import os

import cv2
from ultralytics import YOLO
import config


class VehicleTypeClassifier:
    """Phân loại TINH (fine-grained) loại xe từ crop bbox.

    Dùng model YOLO-cls (yolov8n-cls) đã train riêng cho các dạng xe:
    xe máy, xe tải, xe chở dầu, ô tô con, xe buýt, xe đạp, tàu hỏa...

    CHỐNG CHE KHUẤT / XE KHÔNG HOÀN CHỈNH:
      - Bbox bị che (nhỏ / thiếu góc cạnh) → mở rộng context crop hơn
        (VEHICLE_CLS_CROP_MARGIN_OCCLUDED) và upscale crop nhỏ về 224x224
        để model nhìn thấy đủ thân xe.
      - Temporal MAJORITY VOTE theo track_id: 1 frame phân loại sai/thiếu
        không làm đổi nhãn; cần VOTE_MIN frame ủng hộ CÙNG 1 loại xe mới
        chốt nhãn (ổn định khi xe bị che thoáng qua / lúc đi qua vật cản).

    Trả về tên hiển thị TIẾNG VIỆT (map qua config.VEHICLE_CLS_NAME_MAP).
    Nếu model chưa có (file weights không tồn tại) -> tự vô hiệu hoá,
    pipeline phát hiện/rule vẫn chạy bình thường (giữ cls_id COCO).
    """

    def __init__(self):
        self.enabled = config.VEHICLE_CLS_ENABLED
        self.model = None
        self.names = {}
        self.votes = {}  # track_id -> {label: count} đa số theo cửa sổ trượt
        if self.enabled:
            path = config.VEHICLE_CLS_MODEL_PATH
            if os.path.exists(path):
                try:
                    self.model = YOLO(path)
                    self.names = self.model.names if self.model.names else {}
                    print(f"[VehicleCls] Model phân loại xe: {path}")
                except Exception as e:
                    print(f"[VehicleCls] Lỗi load model phân loại xe: {e}")
                    self.model = None
            else:
                print(f"[VehicleCls] KHÔNG thấy {path} — bỏ qua phân loại tinh.")

    @property
    def available(self):
        return self.model is not None

    def classify(self, frame_bgr, bbox, track_id=None):
        """Phân loại crop xe -> (tên tiếng Việt, conf). Trả (None, 0.0) nếu bỏ qua.

        Nếu truyền track_id, kết quả nằm trong bộ đếm đa số (majority vote)
        theo thời gian → trả nhãn CHỈ khi cửa sổ vote xác nhận.
        """
        if not self.available:
            return None, 0.0

        crop = self._crop_context(frame_bgr, bbox)
        if crop is None:
            return None, 0.0

        try:
            res = self.model(crop, verbose=False)
        except Exception:
            return None, 0.0
        if not res or res[0].probs is None:
            return None, 0.0

        probs = res[0].probs
        top = int(probs.top1)
        conf = float(probs.top1conf)
        raw = self.names.get(top, str(top))
        label = self._to_vietnamese(raw)

        if track_id is None:
            return label, conf

        self._record_vote(track_id, label, conf)
        return self._resolve_vote(track_id)

    # ----------------------------------------------------------------
    # CROP — chống che khuất / xe không hoàn chỉnh
    # ----------------------------------------------------------------
    def _crop_context(self, frame_bgr, bbox):
        """Tạo crop xoay quanh bbox, mở rộng context khi bbox NHỎ (bị che).

        - Bbox lớn: dùng VEHICLE_CLS_CROP_MARGIN (nhẹ).
        - Bbox nhỏ (< 60px cạnh ngắn): mở rộng context hơn (marginal occluded)
          để nhìn thấy phần thân xe còn lộ, sau đó upscale về >= 224px để model
          phân loại ổn định.
        Trả None nếu crop quá nhỏ để làm việc.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        bw0, bh0 = x2 - x1, y2 - y1
        if bw0 < 16 or bh0 < 16:
            return None

        # Mở rộng quanh tâm bbox (giữ trọn thân xe; nhỏ hơn khi bị che)
        m = config.VEHICLE_CLS_CROP_MARGIN
        if min(bw0, bh0) < 60:
            m = max(m, config.VEHICLE_CLS_CROP_MARGIN_OCCLUDED)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = bw0 * m, bh0 * m
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None

        crop = frame_bgr[y1:y2, x1:x2]

        # Upscale crop nhỏ về >= 224px để model cls không mất chi tiết
        ch, cw = crop.shape[:2]
        if min(ch, cw) < 224:
            scale = max(224.0 / ch, 224.0 / cw)
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        return crop

    # ----------------------------------------------------------------
    # MAJORITY VOTE theo track (chống frame độc lập sai khi bị che)
    # ----------------------------------------------------------------
    def _record_vote(self, track_id, label, conf):
        if conf < config.VEHICLE_CLS_MIN_CONF:
            return  # frame phân loại thiếu chắc chắn -> không ghi
        v = self.votes.setdefault(track_id, {})
        v[label] = v.get(label, 0) + 1
        # Cửa sổ trượt: khi vượt VOTE_KEEP, chia 2 toàn bộ để giữ phần mới
        total = sum(v.values())
        if total > config.VEHICLE_CLS_VOTE_KEEP:
            for k in list(v):
                v[k] = (v[k] + 1) // 2
            for k in [k for k, c in v.items() if c == 0]:
                del v[k]

    def _resolve_vote(self, track_id):
        """Chốt nhãn theo đa số trong cửa sổ vote. Trả (None, 0.0) khi chưa đủ."""
        v = self.votes.get(track_id, {})
        if not v:
            return None, 0.0
        total = sum(v.values())
        if total < config.VEHICLE_CLS_VOTE_MIN:
            return None, 0.0
        best = max(v, key=v.get)
        best_c = v[best]
        if best_c < config.VEHICLE_CLS_VOTE_MIN:
            return None, 0.0
        if best_c / total < 0.5:  # cần đa số rõ ràng
            return None, 0.0
        # conf token: ổn định = nhiều frame ủng hộ -> càng chắc chắn
        conf = min(1.0, 0.5 + 0.1 * best_c / config.VEHICLE_CLS_VOTE_MIN)
        return best, conf

    def prune_votes(self, active_track_ids):
        """Xoá vote của các track đã mất dấu."""
        active = set(active_track_ids)
        for tid in [t for t in self.votes if t not in active]:
            del self.votes[tid]

    def _to_vietnamese(self, raw):
        return config.VEHICLE_CLS_NAME_MAP.get(raw, raw)