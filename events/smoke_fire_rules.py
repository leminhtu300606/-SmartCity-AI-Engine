import cv2
import numpy as np
import config


class SmokeFireRules:
    """Phát hiện Khói / Lửa bằng visual pattern analysis cải tiến:

    - Giữ nguyên Color Space (không grayscale) để tận dụng thông tin màu cho fire.
    - Dual HSV range cho Fire (red wraps H=0/180).
    - Morphological operations loại bỏ nhiễu nhỏ.
    - Contour area filtering: chỉ tính vùng đủ lớn.
    - Flicker detection: lửa nhấp nháy, vật thể tĩnh màu ấm thì không.
    - Frame differencing cho Smoke: phân biệt khói mới xuất hiện vs bề mặt xám có sẵn.
    - Temporal persistence + Region growth → Event.
    """

    def __init__(self):
        self.fire_persistence_map = {}   # zone -> persistence_count
        self.smoke_persistence_map = {}
        self.fire_area_history = {}      # zone -> [fire_pixels, ...] cho region growth
        self.prev_fire_mask = {}         # zone -> previous fire mask cho flicker detection
        self.prev_gray = {}              # zone -> previous grayscale ROI cho smoke frame diff

    def analyze_frame(self, frame_bgr, roi_polygons):
        events = []
        if frame_bgr is None:
            return events

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ---- FIRE MASK: Dual HSV Ranges (red wraps around H=0/180) ----
        # Range 1: Deep red → orange (H: 0–15)
        fire_mask_1 = cv2.inRange(hsv,
                                  np.array([0, 130, 200], dtype=np.uint8),
                                  np.array([15, 255, 255], dtype=np.uint8))
        # Range 2: Orange → yellow (H: 15–35)
        fire_mask_2 = cv2.inRange(hsv,
                                  np.array([15, 110, 200], dtype=np.uint8),
                                  np.array([35, 255, 255], dtype=np.uint8))
        # Range 3: Deep red (H: 160–180, wrap-around)
        fire_mask_3 = cv2.inRange(hsv,
                                  np.array([160, 130, 200], dtype=np.uint8),
                                  np.array([180, 255, 255], dtype=np.uint8))
        fire_mask_raw = fire_mask_1 | fire_mask_2 | fire_mask_3

        # Morphological open (loại bỏ nhiễu nhỏ) + close (lấp lỗ nhỏ)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask_raw, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        # ---- SMOKE MASK: Narrow HSV (xám/trắng, saturation rất thấp) ----
        # Chỉ bắt pixel có saturation rất thấp VÀ value trung bình-cao (khói thật)
        # Tránh bắt bề mặt tối (đường nhựa tối) hoặc trắng sáng quá (trời sáng)
        smoke_mask_raw = cv2.inRange(hsv,
                                     np.array([0, 0, 140], dtype=np.uint8),
                                     np.array([180, 45, 225], dtype=np.uint8))
        smoke_mask = cv2.morphologyEx(smoke_mask_raw, cv2.MORPH_OPEN, kernel)

        for roi in roi_polygons:
            if roi.get("event_type") != config.EVENT_TYPE_SMOKE_FIRE:
                continue

            zone_name = roi["name"]
            poly = np.array(roi["polygon"], np.int32)

            # Tạo ROI mask
            roi_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [poly], 255)
            roi_area = cv2.countNonZero(roi_mask)

            # ============================================================
            # FIRE ANALYSIS
            # ============================================================
            fire_in_roi = cv2.bitwise_and(fire_mask, roi_mask)
            fire_pixels = self._count_significant_contours(fire_in_roi)

            # Flicker detection: lửa nhấp nháy frame-to-frame, vật thể tĩnh thì không
            flicker_score = self._compute_flicker(zone_name, fire_in_roi)

            # Fire signal = đủ pixel lớn + (có flicker HOẶC rất nhiều pixel)
            is_fire_signal = (
                fire_pixels > config.SMOKE_FIRE_FIRE_PIXEL_THRESH
                and (flicker_score > config.SMOKE_FIRE_FLICKER_THRESH
                     or fire_pixels > config.SMOKE_FIRE_FIRE_PIXEL_THRESH * 3)
            )

            # Temporal persistence (cap để decay nhanh khi hết tín hiệu)
            if is_fire_signal:
                self.fire_persistence_map[zone_name] = min(
                    config.SMOKE_FIRE_FIRE_PERSIST_THRESH + 2,
                    self.fire_persistence_map.get(zone_name, 0) + 1,
                )
            else:
                self.fire_persistence_map[zone_name] = max(
                    0, self.fire_persistence_map.get(zone_name, 0) - 1
                )

            # Region growth: diện tích lửa tăng dần → tăng confidence
            hist = self.fire_area_history.setdefault(zone_name, [])
            hist.append(fire_pixels)
            if len(hist) > config.SMOKE_FIRE_GROWTH_WINDOW:
                hist.pop(0)
            growth = 0.0
            if (len(hist) >= 3
                    and hist[-1] > hist[-2] > hist[-3]
                    and hist[-1] > config.SMOKE_FIRE_FIRE_PIXEL_THRESH):
                growth = 0.08

            # FIRE EVENT
            if self.fire_persistence_map.get(zone_name, 0) >= config.SMOKE_FIRE_FIRE_PERSIST_THRESH:
                events.append({
                    "event_type": "FIRE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": min(0.98, 0.85 + growth + flicker_score * 0.3),
                    "description": "Phát hiện đám cháy trong vùng quan sát",
                })

            # ============================================================
            # SMOKE ANALYSIS
            # ============================================================
            smoke_in_roi = cv2.bitwise_and(smoke_mask, roi_mask)
            smoke_pixels = self._count_significant_contours(smoke_in_roi)

            # Frame differencing: khói là vùng xám MỚI XUẤT HIỆN
            # Bề mặt xám có sẵn (đường, tường) không tạo frame diff lớn
            smoke_change = self._compute_smoke_change(zone_name, gray, roi_mask)

            # Smoke ratio (% diện tích ROI bị khói phủ)
            smoke_ratio = smoke_pixels / max(roi_area, 1)

            # Smoke signal = đủ pixel + có thay đổi gần đây + phủ đủ diện tích
            is_smoke_signal = (
                smoke_pixels > config.SMOKE_FIRE_SMOKE_PIXEL_THRESH
                and smoke_change > 0.02
                and smoke_ratio > 0.01
            )

            if is_smoke_signal:
                self.smoke_persistence_map[zone_name] = min(
                    config.SMOKE_FIRE_SMOKE_PERSIST_THRESH + 3,
                    self.smoke_persistence_map.get(zone_name, 0) + 1,
                )
            else:
                self.smoke_persistence_map[zone_name] = max(
                    0, self.smoke_persistence_map.get(zone_name, 0) - 1
                )

            # SMOKE EVENT
            if self.smoke_persistence_map.get(zone_name, 0) >= config.SMOKE_FIRE_SMOKE_PERSIST_THRESH:
                events.append({
                    "event_type": "SMOKE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": min(0.92, 0.78 + smoke_change * 2),
                    "description": "Phát hiện khói bất thường",
                })

        return events

    # ----------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------

    def _count_significant_contours(self, mask):
        """Chỉ đếm pixel từ contour có diện tích >= ngưỡng (loại bỏ nhiễu nhỏ)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= config.SMOKE_FIRE_MIN_CONTOUR_AREA:
                total += int(area)
        return total

    def _compute_flicker(self, zone_name, fire_mask_current):
        """Flicker detection: tỷ lệ pixel lửa thay đổi giữa 2 frame liên tiếp.

        Lửa thật nhấp nháy (pixel lửa xuất hiện/biến mất liên tục).
        Vật thể tĩnh màu ấm (đèn đường, xe cam) thì không thay đổi → flicker ≈ 0.
        """
        prev = self.prev_fire_mask.get(zone_name)
        self.prev_fire_mask[zone_name] = fire_mask_current.copy()

        if prev is None or prev.shape != fire_mask_current.shape:
            return 0.0

        # XOR: pixel thay đổi giữa 2 frame
        diff = cv2.bitwise_xor(fire_mask_current, prev)
        changed = cv2.countNonZero(diff)
        total = cv2.countNonZero(fire_mask_current) + cv2.countNonZero(prev)

        if total < 50:
            return 0.0
        return changed / max(total, 1)

    def _compute_smoke_change(self, zone_name, gray_frame, roi_mask):
        """Frame differencing cho smoke: phát hiện vùng xám MỚI xuất hiện.

        Khói tạo ra thay đổi lớn trong frame diff; bề mặt xám có sẵn thì ổn định.
        """
        gray_roi = cv2.bitwise_and(gray_frame, roi_mask)
        prev = self.prev_gray.get(zone_name)
        self.prev_gray[zone_name] = gray_roi.copy()

        if prev is None or prev.shape != gray_roi.shape:
            return 0.0

        # Absolute difference + threshold
        diff = cv2.absdiff(gray_roi, prev)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed = cv2.countNonZero(thresh)
        roi_area = cv2.countNonZero(roi_mask)

        return changed / max(roi_area, 1)