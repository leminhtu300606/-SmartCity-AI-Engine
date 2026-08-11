import cv2
import numpy as np
import config


class SmokeFireRules:
    """Phát hiện LỬA — tối giản điều kiện, giảm tham số.

    Chỉ giữ pipeline cốt lõi:
      STAGE 1 — tách vùng màu lửa (HSV đa dải) + morphological.
      STAGE 2 — đủ pixel + đa sắc (hue spread) + NHẤP NHÁY (flicker).
      STAGE 3 — duy trì đủ số frame (persistence window) -> emit FIRE_DETECTED.

    Loại bỏ khói (chỉ hỗ trợ), grid, small fire, glow, moving fire để giảm
    số tham số và điểm dễ bị nhiễu.
    """

    def __init__(self):
        self.fire_persist_window = {}    # zone -> [0/1, ...]
        self.prev_fire_mask = {}         # zone -> prev fire mask (flicker)
        self._frame_count = 0
        self.alert_cooldown = {}         # key -> frame hết hạn cooldown

    def _alert_ready(self, key):
        return self.alert_cooldown.get(key, -1) <= self._frame_count

    def _mark_alert(self, key):
        self.alert_cooldown[key] = (self._frame_count
                                    + config.SMOKE_FIRE_ALERT_COOLDOWN)

    # ================================================================
    # ORCHESTRATOR
    # ================================================================
    def analyze_frame(self, frame_bgr, roi_polygons, object_bboxes=None):
        """Phát hiện lửa cho từng ROI (zone). object_bboxes = bbox NGƯỜI để
        loại áo quần đỏ/cam (không phải lửa)."""
        events = []
        if frame_bgr is None:
            return events

        self._frame_count += 1
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # STAGE 1: fire mask (kênh MÀU HSV)
        fire_mask = self._detect_fire_mask(hsv)

        for roi in roi_polygons:
            if roi.get("event_type") != config.EVENT_TYPE_SMOKE_FIRE:
                continue

            zone_name = roi["name"]
            poly = np.array(roi["polygon"], np.int32)
            roi_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [poly], 255)

            fire_in_roi = cv2.bitwise_and(fire_mask, roi_mask)

            # STAGE 2: đủ pixel + bản chất lửa (đa sắc + nhấp nháy)
            fire_pixels = self._count_significant_contours(fire_in_roi)
            hue_spread = self._compute_fire_hue_spread(fire_in_roi, hsv)
            flicker = self._compute_flicker(zone_name, fire_in_roi)

            # Loại áo quần người (đỏ/cam)
            person_overlap = self._fire_inside_objects_ratio(
                fire_in_roi, object_bboxes)
            is_object_clothing = (
                object_bboxes
                and person_overlap > config.SMOKE_FIRE_MAX_PERSON_OVERLAP)

            is_fire_signal = (
                fire_pixels > config.SMOKE_FIRE_FIRE_PIXEL_THRESH
                and hue_spread > config.SMOKE_FIRE_FIRE_HUE_CIRC_VAR_THRESH
                and flicker > config.SMOKE_FIRE_FLICKER_THRESH
                and not is_object_clothing
            )

            # STAGE 3: persistence window-based
            window = config.SMOKE_FIRE_FIRE_PERSIST_THRESH + 4
            hist = self.fire_persist_window.setdefault(zone_name, [])
            hist.append(1 if is_fire_signal else 0)
            if len(hist) > window:
                hist.pop(0)
            sustained = sum(hist) >= config.SMOKE_FIRE_FIRE_PERSIST_THRESH

            if sustained and self._alert_ready(zone_name):
                self._mark_alert(zone_name)
                conf = min(0.98, 0.90 + hue_spread * 0.5)
                ev = {
                    "event_type": "FIRE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": conf,
                    "description": f"Phát hiện đám cháy tại khu vực {zone_name}",
                }
                bbox = self._fire_bbox(fire_in_roi, poly)
                if bbox is not None:
                    ev["bbox"] = bbox
                events.append(ev)

        return events

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _detect_fire_mask(self, hsv):
        """Fire mask từ DUAL HSV ranges (red wraps around H=0/180)."""
        m1 = cv2.inRange(hsv, np.array([0, 110, 180], dtype=np.uint8),
                         np.array([15, 255, 255], dtype=np.uint8))
        m2 = cv2.inRange(hsv, np.array([15, 90, 180], dtype=np.uint8),
                         np.array([35, 255, 255], dtype=np.uint8))
        m3 = cv2.inRange(hsv, np.array([160, 110, 180], dtype=np.uint8),
                         np.array([180, 255, 255], dtype=np.uint8))
        raw = m1 | m2 | m3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    def _count_significant_contours(self, mask):
        """Đếm pixel của contour có diện tích >= ngưỡng (loại nhiễu nhỏ)."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= config.SMOKE_FIRE_MIN_CONTOUR_AREA:
                total += int(area)
        return total

    def _compute_flicker(self, zone_name, fire_mask_current):
        """Tỷ lệ pixel lửa thay đổi giữa 2 frame liên tiếp (nhấp nháy)."""
        prev = self.prev_fire_mask.get(zone_name)
        self.prev_fire_mask[zone_name] = fire_mask_current.copy()
        if prev is None or prev.shape != fire_mask_current.shape:
            return 0.0
        diff = cv2.bitwise_xor(fire_mask_current, prev)
        changed = cv2.countNonZero(diff)
        total = (cv2.countNonZero(fire_mask_current)
                 + cv2.countNonZero(prev))
        if total < 20:
            return 0.0
        return changed / max(total, 1)

    def _compute_fire_hue_spread(self, fire_in_roi, hsv):
        """Độ phân tán màu (hue circular variance) trong vùng lửa.

        Lửa thật đa sắc (lõi trắng -> vàng -> cam -> đỏ) => variance cao.
        Vật đỏ đồng nhất (áo, xe, biển) -> variance ~ 0.
        """
        mask = fire_in_roi > 0
        if np.count_nonzero(mask) < config.SMOKE_FIRE_MIN_CONTOUR_AREA:
            return 0.0
        hue = hsv[:, :, 0].astype(np.float32)[mask]
        sat = hsv[:, :, 1].astype(np.float32)[mask]
        color_mask = sat > 60
        if np.count_nonzero(color_mask) < 10:
            return 0.0
        hue = hue[color_mask]
        theta = (hue / 180.0) * (2.0 * np.pi)
        cos_sum = np.mean(np.cos(theta))
        sin_sum = np.mean(np.sin(theta))
        r = float(np.hypot(cos_sum, sin_sum))
        return max(0.0, 1.0 - r)

    def _fire_inside_objects_ratio(self, fire_mask_roi, object_bboxes):
        """Tỷ lệ pixel "lửa" nằm TRONG bbox người (áo đỏ/cam) -> loại."""
        if not object_bboxes or cv2.countNonZero(fire_mask_roi) == 0:
            return 0.0
        h, w = fire_mask_roi.shape[:2]
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        for b in object_bboxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(obj_mask, (x1, y1), (x2, y2), 255, -1)
        overlap = cv2.countNonZero(cv2.bitwise_and(fire_mask_roi, obj_mask))
        total = cv2.countNonZero(fire_mask_roi)
        return overlap / max(total, 1)

    @staticmethod
    def _fire_bbox(fire_in_roi, poly=None):
        """bbox [x1,y1,x2,y2] của vùng lửa; fallback về nguyên ROI."""
        ys, xs = np.where(fire_in_roi > 0)
        if len(xs) == 0:
            if poly is None:
                return None
            return [int(poly[:, 0].min()), int(poly[:, 1].min()),
                    int(poly[:, 0].max()), int(poly[:, 1].max())]
        return [int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max())]