import cv2
import numpy as np
import config


class SmokeFireRules:
    """Phát hiện Khói / Lửa bằng visual pattern analysis cải tiến:

    LỬA (3 bước: Xử lý ảnh → Trích xuất đặc trưng → Phân loại):
    - Xử lý ảnh: tách vùng màu đỏ/cam/vàng đặc trưng (HSV dual-range) + độ sáng
      tương phản cao so với nền tối xung quanh (lửa nổi bật, nền tối thì không).
    - Trích xuất đặc trưng: cạnh sắc không đều (edge irregularity - lửa có đường
      viền răng cưa, đèn/nguồn sáng tròn mượt thì ≈1), độ nhấp nháy frame-to-frame
      và tần số nhấp nháy theo cửa sổ thời gian (đặc thù lửa thật).
    - Phân loại đối tượng: kết hợp các đặc trưng thành điểm phân loại (fire class
      score) so với ngưỡng để xác định LỬA hay vật thể màu tương tự (đèn đường,
      đèn xe, mây chiếu sáng) → giảm cảnh báo sai.
    - Morphological operations loại bỏ nhiễu nhỏ.
    - Contour area filtering: chỉ tính vùng đủ lớn.
    - Temporal persistence + Region growth → Event.

    KHÓI:
    - Nhận biết dạng mây mờ lan tỏa: độ mờ nội tại (softness/transparency) cao,
      hình dạng thay đổi theo thời gian (shape change), area lan tỏa (growth).
    - Frame differencing: phân biệt khói mới xuất hiện vs bề mặt xám có sẵn.
    - Phát hiện khói GIAI ĐOẠN ĐẦU khi chưa thấy rõ ngọn lửa (ngưỡng pixel thấp
      hơn + confirm nhanh hơn) → tăng thời gian phản ứng an toàn.
    """

    def __init__(self):
        self.fire_persistence_map = {}   # zone -> persistence_count
        self.smoke_persistence_map = {}
        self.early_smoke_persistence_map = {}  # zone -> persistence cho khói giai đoạn đầu
        self.fire_area_history = {}      # zone -> [fire_pixels, ...] cho region growth
        self.fire_signal_history = {}    # zone -> [bool, ...] cho flicker frequency
        self.prev_fire_mask = {}         # zone -> previous fire mask cho flicker detection
        self.prev_smoke_mask = {}        # zone -> previous smoke mask cho shape change
        self.prev_gray = {}              # zone -> previous grayscale ROI cho smoke frame diff

    def analyze_frame(self, frame_bgr, roi_polygons):
        events = []
        if frame_bgr is None:
            return events

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ---- FIRE MASK: Dual HSV Ranges (red wraps around H=0/180) ----
        # Dải mở rộng (S/V hạ xuống) để nhạy hơn với lửa nhỏ / lửa mờ
        # Range 1: Deep red → orange (H: 0–15)
        fire_mask_1 = cv2.inRange(hsv,
                                  np.array([0, 110, 180], dtype=np.uint8),
                                  np.array([15, 255, 255], dtype=np.uint8))
        # Range 2: Orange → yellow (H: 15–35)
        fire_mask_2 = cv2.inRange(hsv,
                                  np.array([15, 90, 180], dtype=np.uint8),
                                  np.array([35, 255, 255], dtype=np.uint8))
        # Range 3: Deep red (H: 160–180, wrap-around)
        fire_mask_3 = cv2.inRange(hsv,
                                  np.array([160, 110, 180], dtype=np.uint8),
                                  np.array([180, 255, 255], dtype=np.uint8))
        fire_mask_raw = fire_mask_1 | fire_mask_2 | fire_mask_3

        # Morphological open (loại bỏ nhiễu nhỏ) + close (lấp lỗ nhỏ)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask_raw, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        # ---- SMOKE MASK: Narrow HSV (xám/trắng, saturation rất thấp) ----
        # Dải mở rộng (S cao hơn, V rộng hơn) để nhạy hơn với khói loãng / tối
        smoke_mask_raw = cv2.inRange(hsv,
                                     np.array([0, 0, 130], dtype=np.uint8),
                                     np.array([180, 60, 235], dtype=np.uint8))
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

            # Độ sáng tương phản so với nền tối xung quanh (lửa sáng nổi bật)
            fire_contrast = self._compute_fire_contrast(fire_in_roi, roi_mask, hsv)

            # Tần số nhấp nháy liên tục theo cửa sổ thời gian (đặc thù lửa thật)
            flicker_freq = self._compute_flicker_frequency(
                zone_name, fire_pixels > config.SMOKE_FIRE_FIRE_PIXEL_THRESH
            )

            # Cạnh sắc không đều: lửa có đường viền răng cưa, đèn/nguồn sáng
            # tròn mượt thì compactness ≈ 1 → irregularity ≈ 1.
            edge_irregularity = self._compute_fire_edge_irregularity(fire_in_roi)

            # PHÂN LOẠI ĐỐI TƯỢNG: kết hợp đặc trưng thành điểm lửa (0..1)
            # so với cơ sở mẫu. Đèn đường, đèn xe, mây chiếu sáng sẽ có điểm
            # thấp do không đủ (cạnh sắc + nhấp nháy + tương phản) đồng thời.
            fire_class_score = self._classify_fire(
                fire_contrast, flicker_score, flicker_freq, edge_irregularity
            )

            # Fire signal: đủ pixel + tương phản sáng trên nền tối + CẠNH SẮC
            # răng cưa (đèn tròn mượt bị loại) + (nhấp nháy HOẶC cạnh rất xù xì).
            # Phân loại đối tượng: nguồn sáng tròn (đèn) có edge≈1 -> loại dù có
            # nhấp nháy; lửa thật luôn có cạnh không đều + nhấp nháy.
            edge_ok = edge_irregularity > config.SMOKE_FIRE_FIRE_EDGE_IRREGULARITY_THRESH
            flicker_ok = (flicker_score > config.SMOKE_FIRE_FLICKER_THRESH
                          or flicker_freq > config.SMOKE_FIRE_FLICKER_FREQ_THRESH)
            very_jagged = edge_irregularity > config.SMOKE_FIRE_FIRE_JAGGED_THRESH

            is_fire_signal = (
                fire_pixels > config.SMOKE_FIRE_FIRE_PIXEL_THRESH
                and fire_contrast > config.SMOKE_FIRE_CONTRAST_THRESH
                and edge_ok
                and (flicker_ok or very_jagged)
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
                conf = min(0.98, 0.82 + growth
                           + min(fire_class_score, 1.0) * 0.12
                           + min(fire_contrast / 300.0, 0.06))
                events.append({
                    "event_type": "FIRE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": conf,
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

            # Độ trong suốt / mờ nội tại: khói mờ lan tỏa, không có cạnh sắc
            smoke_softness = self._compute_smoke_softness(smoke_in_roi, gray)

            # Hình dạng thay đổi + lan tỏa theo thời gian (mây mờ đổi dạng liên tục)
            shape_change, spread = self._compute_smoke_shape_change(zone_name, smoke_in_roi)

            # Smoke ratio (% diện tích ROI bị khói phủ)
            smoke_ratio = smoke_pixels / max(roi_area, 1)

            # Smoke signal = đủ pixel + mờ (dạng mây) + đổi hình/lan tỏa + frame diff
            is_smoke_signal = (
                smoke_pixels > config.SMOKE_FIRE_SMOKE_PIXEL_THRESH
                and smoke_softness > config.SMOKE_FIRE_SMOKE_SOFTNESS_THRESH
                and (shape_change > config.SMOKE_FIRE_SMOKE_SHAPE_CHANGE_THRESH
                     or spread > 0.05)
                and (smoke_change > config.SMOKE_FIRE_SMOKE_CHANGE_THRESH
                     or smoke_ratio > 0.02)
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

            # KHÓI GIAI ĐOẠN ĐẦU: mờ, lan tỏa, đổi hình khi CHƯA thấy rõ ngọn lửa.
            # Ngưỡng pixel thấp hơn + confirm nhanh hơn để tăng thời gian phản ứng.
            is_early_smoke_signal = (
                smoke_pixels > config.SMOKE_FIRE_EARLY_SMOKE_PIXEL_THRESH
                and smoke_softness > config.SMOKE_FIRE_SMOKE_SOFTNESS_THRESH
                and (shape_change > config.SMOKE_FIRE_SMOKE_SHAPE_CHANGE_THRESH
                     or spread > 0.03)
                and (smoke_change > config.SMOKE_FIRE_SMOKE_CHANGE_THRESH * 0.5
                     or smoke_ratio > 0.01)
            )
            if is_early_smoke_signal:
                self.early_smoke_persistence_map[zone_name] = min(
                    config.SMOKE_FIRE_EARLY_SMOKE_PERSIST_THRESH + 2,
                    self.early_smoke_persistence_map.get(zone_name, 0) + 1,
                )
            else:
                self.early_smoke_persistence_map[zone_name] = max(
                    0, self.early_smoke_persistence_map.get(zone_name, 0) - 1
                )

            # SMOKE EVENT — ưu tiên cảnh báo khói sớm (trước khi thấy lửa)
            if (self.early_smoke_persistence_map.get(zone_name, 0)
                    >= config.SMOKE_FIRE_EARLY_SMOKE_PERSIST_THRESH):
                events.append({
                    "event_type": "SMOKE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": min(0.90, 0.70 + smoke_change * 2
                                      + smoke_softness * 0.2 + spread * 0.5),
                    "description": "Phát hiện khói giai đoạn đầu (chưa thấy rõ lửa)",
                })
            elif (self.smoke_persistence_map.get(zone_name, 0)
                    >= config.SMOKE_FIRE_SMOKE_PERSIST_THRESH):
                events.append({
                    "event_type": "SMOKE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": min(0.92, 0.78 + smoke_change * 2
                                      + smoke_softness * 0.2 + spread * 0.5),
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

    def _compute_fire_contrast(self, fire_in_roi, roi_mask, hsv):
        """Độ sáng tương phản của lửa so với nền tối xung quanh.

        Lửa thật rất sáng (V cao) nổi bật trên nền tối. Vật thể ấm sáng trên nền
        sáng (đường, trời sáng) có độ tương phản thấp → bị loại.
        """
        fire_px = cv2.countNonZero(fire_in_roi)
        if fire_px < config.SMOKE_FIRE_MIN_CONTOUR_AREA:
            return 0.0

        # Vùng nền xung quanh lửa (dilate fire, loại bỏ pixel lửa)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        surround = cv2.dilate(fire_in_roi, kernel)
        surround = cv2.bitwise_and(surround, cv2.bitwise_not(fire_in_roi))
        surround = cv2.bitwise_and(surround, roi_mask)

        v_channel = hsv[:, :, 2].astype(np.float32)

        fire_vals = v_channel[fire_in_roi > 0]
        bg_vals = v_channel[surround > 0]
        if fire_vals.size == 0 or bg_vals.size == 0:
            return 0.0

        fire_mean = float(fire_vals.mean())
        bg_mean = float(bg_vals.mean())
        return fire_mean - bg_mean

    def _compute_flicker_frequency(self, zone_name, is_fire_present):
        """Tần số nhấp nháy liên tục theo cửa sổ thời gian.

        Lửa thật dao động bật/tắt liên tục giữa các frame → tỷ lệ thay đổi trạng
        thái trong cửa sổ cao. Vật thể tĩnh màu ấm luôn present → tần số ≈ 0.
        """
        hist = self.fire_signal_history.setdefault(zone_name, [])
        hist.append(1 if is_fire_present else 0)
        if len(hist) > config.SMOKE_FIRE_FLICKER_FREQ_WINDOW:
            hist.pop(0)
        if len(hist) < max(2, config.SMOKE_FIRE_FLICKER_FREQ_WINDOW // 2):
            return 0.0

        transitions = sum(1 for i in range(1, len(hist)) if hist[i] != hist[i - 1])
        return transitions / float(len(hist) - 1)

    def _compute_fire_edge_irregularity(self, fire_in_roi):
        """Cạnh sắc không đều của vùng lửa (edge irregularity / jaggedness).

        Dùng compactness = 4*pi*A / P². Hình tròn mượt (đèn, nguồn sáng) → ≈1.
        Lửa thật có đường viền răng cưa, xù xì → tỷ lệ nghịch (irregularity > 1).
        Lấy giá trị cao nhất trong các contour đủ lớn (ngọn lửa chính).
        """
        contours, _ = cv2.findContours(fire_in_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        irregularity = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.SMOKE_FIRE_MIN_CONTOUR_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            compactness = (4.0 * np.pi * area) / (perimeter * perimeter)
            if compactness > 0:
                irregularity = max(irregularity, 1.0 / compactness)
        return irregularity

    def _classify_fire(self, fire_contrast, flicker_score, flicker_freq, edge_irregularity):
        """Phân loại đối tượng: kết hợp đặc trưng thành điểm LỬA (0..1).

        So sánh với cơ sở mẫu lửa thật: phải ĐỒNG THỜI có màu sáng nổi bật trên
        nền tối (contrast), nhấp nháy (flicker) và cạnh sắc không đều. Vật thể
        màu tương tự (đèn, mây) chỉ đạt 1-2 đặc trưng → điểm thấp.
        """
        # Chuẩn hoá từng đặc trưng về [0..1]
        norm_contrast = min(max(fire_contrast / 150.0, 0.0), 1.0)
        norm_flicker = min(max(flicker_score / 0.4, 0.0), 1.0)
        norm_freq = min(max(flicker_freq / 0.6, 0.0), 1.0)
        norm_edge = min(max((edge_irregularity - 1.0) / 1.5, 0.0), 1.0)

        # Lửa thật: contrast CAO (bắt buộc) + cạnh sắc răng cưa + nhấp nháy.
        # Điểm thấp nếu thiếu 1 trong các đặc trưng trọng yếu.
        score = (
            norm_contrast * 0.25
            + norm_flicker * 0.25
            + norm_freq * 0.20
            + norm_edge * 0.30
        )
        return score

    def _compute_smoke_softness(self, smoke_in_roi, gray_frame):
        """Độ trong suốt/mờ nội tại của vùng khói.

        Khói là mây mờ lan tỏa: nội tại gradient thấp, không có cạnh sắc.
        Đối tượng đặc (xe, tường, người) có cạnh/chi tiết → gradient cao.
        """
        if cv2.countNonZero(smoke_in_roi) < config.SMOKE_FIRE_MIN_CONTOUR_AREA:
            return 0.0

        gx = cv2.Sobel(gray_frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_frame, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        smoke_px = smoke_in_roi > 0
        if np.count_nonzero(smoke_px) == 0:
            return 0.0
        vals = mag[smoke_px]
        soft = float(np.mean(vals < config.SMOKE_FIRE_SMOKE_SOFT_GRAD_THRESH))
        return soft

    def _compute_smoke_shape_change(self, zone_name, smoke_mask_current):
        """Hình dạng khói thay đổi + độ lan tỏa giữa 2 frame.

        Mây mờ lan tỏa liên tục đổi hình dạng: tỷ lệ pixel XOR thay đổi lớn
        (shape change) và diện tích tăng dần (spread). Khói ổn định/tĩnh thì ≈ 0.
        """
        prev = self.prev_smoke_mask.get(zone_name)
        self.prev_smoke_mask[zone_name] = smoke_mask_current.copy()

        if prev is None or prev.shape != smoke_mask_current.shape:
            return 0.0, 0.0

        cur = cv2.countNonZero(smoke_mask_current)
        old = cv2.countNonZero(prev)
        if cur + old < config.SMOKE_FIRE_MIN_CONTOUR_AREA:
            return 0.0, 0.0

        xor = cv2.bitwise_xor(smoke_mask_current, prev)
        changed = cv2.countNonZero(xor)
        shape_change = changed / max(cur + old, 1)

        # Lan tỏa: khói mới > khói cũ → diện tích tăng dần theo thời gian
        spread = (cur - old) / max(old, 1)
        spread = max(0.0, spread)

        return shape_change, spread