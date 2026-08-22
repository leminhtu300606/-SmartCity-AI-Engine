"""CameraContext — state PER-CAMERA cho AI pipeline.

Rule 2: Camera capture chỉ ghi LATEST frame vào đây (1 frame mới nhất,
ghi đè frame cũ). Rule 4: không queue vô hạn.
Mọi model nằm trong inference/registry (SHARED), context KHÔNG giữ model.

Trạng thái per-camera (KHÔNG model): memory_mgr, classifier, alert keys,
cadence timers — đúng Rule 10 (state per camera, model shared).
"""
import threading
import time
from typing import Any

import config
from tracker.memory_manager import ObjectMemoryManager


class CameraContext:
    def __init__(self, camera_id: str, stream_url: str):
        self.camera_id = camera_id
        self.stream_url = stream_url

        # ---- Latest frame slot (Rule 2/3) ----
        self._lock = threading.Lock()
        self.latest_frame: Any = None         # frame BGR đã resize
        self.latest_idx = 0              # frame index
        self.latest_ts = 0.0             # timestamp camera
        self.has_new = False

        # ---- AI state per camera (shared models ở registry) ----
        self.memory_mgr = ObjectMemoryManager(maxlen=config.TEMPORAL_BUFFER_MAXLEN)
        self.active_alert_keys: set = set()
        self.last_annotated: Any = None   # frame AI đã vẽ overlay (GUI hiển thị)
        self.annotated_version = 0   # tăng mỗi lần AI vẽ xong frame mới
        self._ai_last_processed = 0

        # ---- Cadence timers (time-based, Rule 3) ----
        self.next_detect_at = 0.0
        self.next_fire_at = 0.0
        self.next_accident_at = 0.0

        # ---- Thống kê cho benchmark ----
        self.detect_count = 0
        self._missed_cadence = 0
        self.start_time = time.time()

    # ------------------------------------------------------------
    # Camera thread gọi — CAPTURE CHỈ, không model
    # ------------------------------------------------------------
    def push_frame(self, frame_bgr: Any, frame_idx: int) -> None:
        """Camera capture: ghi đè LATEST frame, bỏ frame cũ (1 frame).

        Camera FPS (25-30) != AI FPS (5): bỏ frame cũ là INTENDED (Rule 3),
        không tính là drop. Drop thật = khi AI không kịp cadence (đo ở scheduler).
        """
        with self._lock:
            self.latest_frame = frame_bgr
            self.latest_idx = frame_idx
            self.latest_ts = time.time()
            self.has_new = True

    def record_missed_cadence(self) -> None:
        """Scheduler gọi khi 1 tick detect bị bỏ vì job trước chưa xong
        (AI không theo kịp cadence — đây mới là 'frame drop' thật)."""
        with self._lock:
            self._missed_cadence += 1

    def mark_ai_processed(self, idx: int) -> None:
        with self._lock:
            self._ai_last_processed = idx

    def take_latest(self) -> tuple[Any, int, float]:
        """Scheduler lấy frame mới nhất (chỉ 1 frame — không queue vô hạn)."""
        with self._lock:
            if not self.has_new:
                return None, -1, 0.0
            frame = self.latest_frame
            idx = self.latest_idx
            self.has_new = False
            return frame, idx, self.latest_ts

    @property
    def has_new_frame(self) -> bool:
        with self._lock:
            return self.has_new

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    def fps(self) -> float:
        dt = max(time.time() - self.start_time, 1e-5)
        return self.detect_count / dt

    def drop_rate(self) -> float:
        dt = max(self.detect_count + self._missed_cadence, 1)
        return self._missed_cadence / dt
