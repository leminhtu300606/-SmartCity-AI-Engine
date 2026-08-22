"""EmbeddingPipeline — Rule 9: Embedding tách khỏi inference loop.

Detection event chạy xong → crop gửi vào queue → worker riêng chạy
embedding (MobileNetV3 feature extractor) → FAISS. Embedding KHÔNG bao giờ
chặn detection (không nằm critical path).

Queue giới hạn (config.EMBED_QUEUE_MAXSIZE) — đầy thì bỏ, không tạo
queue vô hạn (Rule 4).

Model là SHARED singleton (Rule 1): một pipeline dùng chung cho mọi camera;
state phân biệt qua metadata camera_id/event_type.
"""
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

import config

# Guard imports để model thiếu weights / thiếu faiss vẫn chạy bình thường
try:
    import faiss  # noqa: F401
    _FAISS_OK = True
except Exception:  # pragma: no cover
    _FAISS_OK = False

try:
    import torch
    _TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_OK = False


class EmbeddingPipeline:
    """MobileNetV3 (feature) → FAISS, chạy async trên worker riêng.

    - Không có torchvision → no-op (queue vẫn hoạt động, không chặn loop).
    - Không có faiss → index in-memory numpy (vẫn lưu vector để benchmark).
    """

    def __init__(self, feature_dim=None, maxsize=None):
        self.dim = int(feature_dim or config.EMBEDDING_FEATURE_DIM)
        self.maxsize = maxsize or config.EMBED_QUEUE_MAXSIZE
        self.enabled = bool(getattr(config, "EMBEDDING_ENABLED", True))

        self.queue = deque(maxlen=self.maxsize)
        self._lock = threading.Lock()
        self._stopped = False
        self._enqueued = 0
        self._processed = 0
        self._dropped = 0
        self._model = None
        self._index = None
        self._n_count = 0
        self._project = None  # random projection nếu dim model != dim cấu hình
        self._feature_buf = None  # tensor tái sử dụng (tránh alloc mỗi event)

        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="embedding-worker")
        self._started = False
        self._log_count = 0
        self._load_resources()

    # ------------------------------------------------------------
    # Khởi tạo resources (KHÔNG chạy trong critical path; worker riêng)
    # ------------------------------------------------------------
    def _load_resources(self):
        if not self.enabled:
            print("[Embedding] EMBEDDING_ENABLED=False — embedding tắt.")
            return
        model_dim = self._try_load_mobilenet()
        self._prepare_index(model_dim)

    def _try_load_mobilenet(self):
        """Load MobileNetV3 feature extractor (torchvision). Trả dim feature.

        Thứ tự weights: file EMBEDDING_MODEL_PATH nếu có → pretrained DEFAULT
        (tự tải về) → None (random). Thiếu torchvision → return 0 (no-op).
        """
        try:
            import torchvision.models as tvm
        except Exception:
            print("[Embedding] Thiếu torchvision — embedding no-op "
                  "(sẽ không chặn pipeline).")
            return 0

        model = None
        source = ""
        path = getattr(config, "EMBEDDING_MODEL_PATH", "") or ""
        try:
            if path and os.path.exists(path):
                model = tvm.mobilenet_v3_large()
                state = torch.load(path, map_location="cpu",
                                   weights_only=True)
                if "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                source = f"file {path}"
        except Exception as e:
            print(f"[Embedding] Load weights từ {path} thất bại: {e}")

        if model is None:
            try:
                weights = tvm.MobileNet_V3_Large_Weights.DEFAULT
                model = tvm.mobilenet_v3_large(weights=weights)
                source = "torchvision pretrained"
            except Exception:
                try:
                    model = tvm.mobilenet_v3_large()
                    source = "torchvision (random weights)"
                except Exception:
                    model = None

        if model is None:
            print("[Embedding] Không dựng được MobileNetV3 — embedding no-op.")
            return 0
        if config.AI_MAX_THREADS > 0:
            try:
                torch.set_num_threads(int(config.AI_MAX_THREADS))
            except Exception:
                pass
        model.eval()
        self._model = model
        print(f"[Embedding] MobileNetV3 feature extractor sẵn sàng ({source}).")

        # MobileNetV3-Large: features output 960D sau global pooling
        try:
            probe = np.zeros((1, 3, 224, 224), dtype=np.float32)
            outp = self._model.features(
                torch.from_numpy(probe)).mean(dim=(2, 3))
            return int(outp.shape[1])
        except Exception:
            return 960

    def _prepare_index(self, model_dim):
        feature_dim = self.dim if self.dim else model_dim
        if feature_dim <= 0:
            return
        if model_dim > 0 and feature_dim != model_dim:
            # projection model_dim -> feature_dim; dùng vec @ _project
            rng = np.random.RandomState(42)
            self._project = rng.normal(
                size=(model_dim, feature_dim)).astype(np.float32)
        path = getattr(config, "EMBEDDING_FAISS_INDEX", "") or ""
        if _FAISS_OK and path and os.path.exists(path):
            try:
                self._index = faiss.read_index(path)
                self._n_count = self._index.ntotal
                print(f"[Embedding] FAISS index: {path} ({self._n_count} vectors).")
                return
            except Exception as e:
                print(f"[Embedding] Đọc FAISS index {path} lỗi: {e}")
        if _FAISS_OK:
            self._index = faiss.IndexFlatIP(feature_dim)
            print(f"[Embedding] FAISS in-memory index (IP, dim={feature_dim}).")
        else:
            print("[Embedding] Không có faiss — dùng index in-memory numpy.")

    # ------------------------------------------------------------
    # Camera loop gọi — ASYNC: chỉ enqueue, không chạy model
    # ------------------------------------------------------------
    def submit(self, crop_bgr, metadata=None):
        """Gửi crop vào queue async. Trả False nếu queue đã đầy (bỏ frame)."""
        if not self.enabled:
            return False
        if crop_bgr is None:
            return False
        with self._lock:
            if len(self.queue) >= self.maxsize:
                self._dropped += 1
                return False
            self.queue.append((crop_bgr, metadata))
            self._enqueued += 1
        self.start()
        return True

    # ------------------------------------------------------------
    # Worker thread — embedding KHÔNG nằm trong detection loop
    # ------------------------------------------------------------
    def start(self):
        """Khởi động worker thread (1 lần). Gọi lười ở submit đầu tiên."""
        if not self._started and not self._stopped:
            self._worker.start()
            self._started = True

    def stop(self):
        self._stopped = True

    def _run(self):
        while not self._stopped:
            item = None
            with self._lock:
                if self.queue:
                    item = self.queue.popleft()
            if item is None:
                time.sleep(0.05)
                continue
            crop, meta = item
            try:
                self._embed(crop, meta)
            except Exception as e:
                print(f"[Embedding] Lỗi xử lý crop: {e}")
            finally:
                with self._lock:
                    self._processed += 1

    def _embed(self, crop_bgr, meta):
        if self._model is None:
            return
        vec = self._extract_feature(crop_bgr)
        if vec is None:
            return
        cam = (meta or {}).get("camera_id", "?")
        evt = (meta or {}).get("event_type", "?")
        if self._index is not None and _FAISS_OK:
            faiss.normalize_L2(vec.reshape(1, -1))
            self._index.add(vec.reshape(1, -1))
            self._n_count += 1
        else:
            self._n_count += 1
        if self._log_count < 3:
            self._log_count += 1
            print(f"[Embedding] {cam}/{evt} → vector {self.dim}D "
                  f"(index={self._n_count})")

    # ----------------------------------------------------------------
    # Preprocess + forward (chạy trong worker; KHÔNG trong detection loop)
    # ----------------------------------------------------------------
    @staticmethod
    def _to_rgb_tensor(crop_bgr):
        im = cv2.resize(crop_bgr, (224, 224))
        arr = im[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2, 0, 1)[None]).contiguous()

    def _extract_feature(self, crop_bgr):
        if not _TORCH_OK:
            return None
        try:
            if self._feature_buf is None:
                self._feature_buf = self._to_rgb_tensor(crop_bgr)
            else:
                self._feature_buf.copy_(self._to_rgb_tensor(crop_bgr))
            with torch.no_grad():
                feats = self._model.features(
                    self._feature_buf).mean(dim=(2, 3))
            vec = feats.squeeze(0).cpu().numpy()
            if self._project is not None:
                vec = vec @ self._project
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                return None
            return (vec / norm).astype(np.float32)
        except Exception:
            return None