import cv2
import numpy as np
import config


class SmokeFireRules:
    """Phát hiện Khói / Lửa theo pipeline VISUAL PATTERN rõ ràng:

    STAGE 1 — VISUAL DETECTION (GIỮ MÀU HSV, không grayscale toàn bộ):
        Tách vùng màu đặc trưng:
          - Lửa : đỏ/cam/vàng (HSV dual-range wrap-around) + độ sáng tương
            phản trên nền tối. Dựa hoàn toàn trên kênh MÀU.
          - Khói: xám/trắng, saturation rất thấp (S thấp, V trung bình).
        + Morphological operations loại nhiễu + contour filtering.

    STAGE 2 — TEMPORAL PERSISTENCE:
        Vùng nghi vấn phải xuất hiện liên tục >= N frame (persistence map)
        trước khi được công nhận.

    STAGE 3 — REGION GROWTH / MOTION:
        Diện tích tăng dần (region growth) + nhấp nháy theo cửa sổ thời gian
        (flicker frequency) + lan tỏa/đổi hình (shape change / spread).
        Phân biệt cháy/khói THẬT với vật tĩnh màu tương tự (đèn, mây, vật đỏ).

    STAGE 4 — FIRE / SMOKE EVENT:
        Đủ tín hiệu (detection + persistence + growth/motion) → emit event
        kèm bbox vị trí (ô grid hoặc zone).

    Toàn bộ feature extraction dùng kênh MÀU (HSV) cho lửa; grayscale chỉ
    dùng phụ trợ riêng cho khói (softness / frame diff) chứ không grayscale
    toàn pipeline — thông tin màu luôn được giữ cho fire detection.
    """

    def __init__(self):
        # STAGE 2 state: zone -> persistence counter
        self.fire_persistence_map = {}
        self.smoke_persistence_map = {}
        # STAGE 3 state: zone -> history (region growth, flicker frequency)
        self.fire_area_history = {}      # zone -> [fire_pixels, ...]
        self.fire_signal_history = {}    # zone -> [bool, ...] cho flicker freq
        self.prev_fire_mask = {}         # zone -> prev fire mask (flicker)
        self.prev_smoke_mask = {}        # zone -> prev smoke mask (shape change)
        self.prev_gray = {}              # zone -> prev grayscale ROI (smoke diff)

        # Lửa nhỏ DI ĐỘNG (nguồn sáng nhỏ không cố định): tín hiệu gom ở mức ROI.
        # Vì điểm lửa nhỏ hay di chuyển giữa các ô grid, persistence theo từng ô
        # không bao giờ đủ frame liên tiếp → gom tín hiệu ở mức ROI.
        self.moving_fire_counter = {}    # "move|<roi_base>" -> sustained_count
        self.moving_fire_cells = {}      # "move|<roi_base>" -> [(cell_keys, bboxes), ...]
        self._frame_cell_signals = {}    # roi_base -> set(cell_name) trong frame hiện tại
        self._frame_cell_bboxes = {}     # roi_base -> [bbox, ...] trong frame hiện tại

        # COOLDOWN: frame_count -> thời điểm được phép báo lại từng zone.
        # Sau khi emit event, zone bị "cấm" trong SMOKE_FIRE_ALERT_COOLDOWN frame
        # để diệt tình trạng cùng 1 ô báo lặp lại liên tục (false positive kéo dài).
        self._frame_count = 0
        self.alert_cooldown = {}         # key -> frame đủ hạn được báo lại

    def _alert_ready(self, key):
        """True nếu zone `key` hết cooldown (được phép báo event mới)."""
        return self.alert_cooldown.get(key, -1) <= self._frame_count

    def _mark_alert(self, key):
        """Đánh dấu zone `key` vừa báo → cấm báo lại trong N frame."""
        self.alert_cooldown[key] = (self._frame_count
                                    + config.SMOKE_FIRE_ALERT_COOLDOWN)

    # ================================================================
    # ORCHESTRATOR — chạy pipeline cho từng ROI / ô grid
    # ================================================================
    def analyze_frame(self, frame_bgr, roi_polygons, object_bboxes=None):
        """Phát hiện khói/lửa theo TỪNG Ô NHỎ (grid) trong khung hình.

        Mỗi ROI được chia thành lưới ô nhỏ (SMOKE_FIRE_GRID_COLS x ROWS). Mỗi ô
        được phân tích ĐỘC LẬP (persistence riêng theo zone_name của ô) nên:
        - Đám cháy/khói cục bộ nhỏ bị một ô "kẹp" chặt → không bị pha loãng bởi
          toàn frame.
        - Event trả về kèm bbox của ô → định vị chính xác vị trí nghi vấn.
        Nếu ô nhỏ quá (không đủ pixel), mạng lưới vẫn giữ ngưỡng tỉ lệ để nhạy.

        object_bboxes: list [x1, y1, x2, y2] của NGƯỜI (person) đã detect bởi
        YOLO. Vùng "lửa" nằm chủ yếu trong bbox người = áo quần màu đỏ/cam →
        bị loại (không phải lửa thật).
        """
        events = []
        if frame_bgr is None:
            return events

        self._frame_count += 1

        # Reset tín hiệu frame hiện tại (gom lửa nhỏ theo ROI base)
        self._frame_cell_signals = {}
        self._frame_cell_bboxes = {}

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # ---- STAGE 1: VISUAL DETECTION — fire mask (kênh MÀU HSV) ----
        fire_mask = self._detect_fire_mask(hsv)
        # ---- STAGE 1: VISUAL DETECTION — smoke mask (S thấp, xám/trắng) ----
        smoke_mask = self._detect_smoke_mask(hsv)
        # ---- STAGE 1: quầng sáng ấm (lửa bị che khuất sau vật cản) ----
        glow_mask = (self._detect_fire_glow_mask(hsv, fire_mask)
                     if config.SMOKE_FIRE_GLOW_ENABLED else None)

        for roi in roi_polygons:
            if roi.get("event_type") != config.EVENT_TYPE_SMOKE_FIRE:
                continue

            base_name = roi["name"]
            poly = np.array(roi["polygon"], np.int32)

            # Tạo ROI mask
            roi_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [poly], 255)

            # Chia ROI thành lưới ô nhỏ; mỗi ô là 1 zone độc lập.
            if config.SMOKE_FIRE_GRID_ENABLED:
                events.extend(self._analyze_grid(
                    base_name, roi_mask, hsv, fire_mask, smoke_mask,
                    glow_mask=glow_mask, object_bboxes=object_bboxes,
                ))
            else:
                events.extend(self._analyze_zone(
                    base_name, roi_mask, None, hsv, fire_mask, smoke_mask,
                    glow_mask=glow_mask, object_bboxes=object_bboxes,
                ))

            # Lửa nhỏ DI ĐỘNG: gom tín hiệu từ các ô trong ROI này
            moving_ev = self._aggregate_moving_fire(
                base_name,
                self._frame_cell_signals.get(base_name, set()),
                self._frame_cell_bboxes.get(base_name, []),
                object_bboxes=object_bboxes,
            )
            if moving_ev is not None:
                events.append(moving_ev)

        return events

    # ================================================================
    # STAGE 1 — VISUAL DETECTION (giữ màu HSV)
    # ================================================================
    def _detect_fire_mask(self, hsv):
        """Fire mask từ DUAL HSV ranges (red wraps around H=0/180).

        Dải mở rộng (S/V hạ xuống) để nhạy hơn với lửa nhỏ / lửa mờ.
        KHÔNG grayscale — dựa trên kênh màu HSV.
        """
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
        # Kernel OPEN hạ từ 5x5 → 3x3: giữ lại blob lửa NHỎ (vài chục pixel)
        # không bị xoá sạch; nhiễu specks 1-2px vẫn bị loại bỏ.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fire_mask = cv2.morphologyEx(fire_mask_raw, cv2.MORPH_OPEN, kernel)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel_close)
        return fire_mask

    def _detect_fire_glow_mask(self, hsv, fire_mask=None):
        """Quầng sáng ấm (glow) — dấu hiệu lửa bị che khuất sau vật cản.

        Ngọn lửa ẩn sau tường/container/bờ kè vẫn hắt ánh cam MỜ lên bề mặt
        xung quanh (S thấp hơn và V trung bình hơn lửa nhìn trực tiếp, vì là
        ánh phản chiếu/chiếu xuyên). Dải HSV mở rộng về phía S thấp / V vừa
        để bắt phần quầng này; trừ đi fire_mask để chỉ giữ quầng, không trùng
        với vùng lửa nhìn trực tiếp (vùng đó do pipeline fire chính xử lý).
        """
        glow_mask_1 = cv2.inRange(hsv,
                                  np.array([0, 40, 90], dtype=np.uint8),
                                  np.array([40, 160, 255], dtype=np.uint8))
        glow_mask_2 = cv2.inRange(hsv,
                                  np.array([160, 40, 90], dtype=np.uint8),
                                  np.array([180, 160, 255], dtype=np.uint8))
        glow_raw = glow_mask_1 | glow_mask_2
        if fire_mask is not None:
            glow_raw = cv2.bitwise_and(glow_raw, cv2.bitwise_not(fire_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glow = cv2.morphologyEx(glow_raw, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(glow, cv2.MORPH_CLOSE, kernel)

    def _detect_smoke_mask(self, hsv):
        """Smoke mask từ HSV hẹp (xám/trắng, saturation rất thấp).

        Dải mở rộng (S cao hơn, V rộng hơn) để nhạy hơn với khói loãng / tối.
        """
        smoke_mask_raw = cv2.inRange(hsv,
                                     np.array([0, 0, 130], dtype=np.uint8),
                                     np.array([180, 60, 235], dtype=np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(smoke_mask_raw, cv2.MORPH_OPEN, kernel)

    def _analyze_grid(self, base_name, roi_mask, hsv, fire_mask, smoke_mask,
                      glow_mask=None, object_bboxes=None):
        """Chia ROI thành lưới ô nhỏ và phân tích từng ô riêng biệt.

        Mỗi ô có zone_name = f"{base_name}|r{r}c{c}" → persistence theo ô.
        Event trả về kèm "bbox": [x1, y1, x2, y2] vị trí ô trên khung hình.
        Ngưỡng pixel/contour được nhân theo hệ số ô (nhỏ hơn toàn frame) để
        vẫn bắt được hiện tượng cục bộ bên trong một ô.
        """
        events = []
        h, w = roi_mask.shape[:2]
        cols = config.SMOKE_FIRE_GRID_COLS
        rows = config.SMOKE_FIRE_GRID_ROWS

        cell_w = max(w // cols, 1)
        cell_h = max(h // rows, 1)

        # Ngưỡng theo ô nhỏ
        fire_px_thresh = int(config.SMOKE_FIRE_FIRE_PIXEL_THRESH
                             * config.SMOKE_FIRE_GRID_FIRE_PIXEL_FACTOR)
        smoke_px_thresh = int(config.SMOKE_FIRE_SMOKE_PIXEL_THRESH
                              * config.SMOKE_FIRE_GRID_SMOKE_PIXEL_FACTOR)

        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * cell_w, r * cell_h
                x2 = min(w, (c + 1) * cell_w)
                y2 = min(h, (r + 1) * cell_h)
                if x2 <= x1 or y2 <= y1:
                    continue

                cell_name = f"{base_name}|r{r}c{c}"

                cell_mask = np.zeros_like(roi_mask)
                cell_mask[y1:y2, x1:x2] = cv2.bitwise_and(
                    roi_mask[y1:y2, x1:x2], 255
                )

                cell_events = self._analyze_zone(
                    cell_name, cell_mask,
                    [x1, y1, x2, y2],
                    hsv, fire_mask, smoke_mask,
                    fire_px_thresh=fire_px_thresh,
                    smoke_px_thresh=smoke_px_thresh,
                    glow_mask=glow_mask,
                    object_bboxes=object_bboxes,
                )
                events.extend(cell_events)

        return events

    # ================================================================
    # PIPELINE CHÍNH — STAGE 1→2→3→4 cho MỘT zone / ô
    # ================================================================
    def _analyze_zone(self, zone_name, roi_mask, bbox, hsv,
                      fire_mask, smoke_mask,
                      fire_px_thresh=None, smoke_px_thresh=None,
                      glow_mask=None, object_bboxes=None):
        """Chạy pipeline 4 stage cho một vùng (ROI hoặc ô grid)."""
        if fire_px_thresh is None:
            fire_px_thresh = config.SMOKE_FIRE_FIRE_PIXEL_THRESH
        if smoke_px_thresh is None:
            smoke_px_thresh = config.SMOKE_FIRE_SMOKE_PIXEL_THRESH
        min_contour = int(config.SMOKE_FIRE_MIN_CONTOUR_AREA
                          * config.SMOKE_FIRE_GRID_MIN_CONTOUR_FACTOR) \
            if fire_px_thresh != config.SMOKE_FIRE_FIRE_PIXEL_THRESH \
            else config.SMOKE_FIRE_MIN_CONTOUR_AREA

        events = []
        roi_area = cv2.countNonZero(roi_mask)

        # ==========================================================
        # FIRE PIPELINE — lửa nhìn trực tiếp
        # ==========================================================
        fire_ev = self._fire_pipeline(
            zone_name, roi_mask, bbox, hsv, fire_mask,
            roi_area, fire_px_thresh, min_contour,
            object_bboxes=object_bboxes,
        )
        has_direct_fire = fire_ev is not None
        if fire_ev is not None:
            events.append(fire_ev)

        # ==========================================================
        # SMALL FIRE — điểm cháy nhỏ / lửa chỉ lộ một phần (chỉ khi
        # chưa có lửa trực tiếp ở zone này, tránh báo trùng).
        # ==========================================================
        if config.SMOKE_FIRE_SMALL_FIRE_ENABLED and not has_direct_fire:
            small_ev = self._small_fire_pipeline(
                zone_name, roi_mask, bbox, hsv, fire_mask,
                config.SMOKE_FIRE_SMALL_FIRE_PIXEL_THRESH,
                config.SMOKE_FIRE_SMALL_FIRE_MIN_CONTOUR,
                object_bboxes=object_bboxes,
            )
            if small_ev is not None:
                events.append(small_ev)
                has_direct_fire = True

        # ==========================================================
        # FIRE GLOW — lửa bị che khuất hoàn toàn (tường/container):
        # bắt quầng sáng ấm nhấp nháy. Chỉ khi chưa có tín hiệu lửa.
        # ==========================================================
        if (config.SMOKE_FIRE_GLOW_ENABLED and glow_mask is not None
                and not has_direct_fire):
            glow_ev = self._fire_glow_pipeline(
                zone_name, roi_mask, bbox, hsv, glow_mask,
                config.SMOKE_FIRE_GLOW_MIN_PIXELS,
                config.SMOKE_FIRE_GLOW_MIN_CONTOUR,
                object_bboxes=object_bboxes,
            )
            if glow_ev is not None:
                events.append(glow_ev)

        # ==========================================================
        # SMOKE PIPELINE
        # ==========================================================
        smoke_ev = self._smoke_pipeline(
            zone_name, roi_mask, bbox, hsv, smoke_mask,
            roi_area, smoke_px_thresh, min_contour,
        )
        if smoke_ev is not None:
            events.append(smoke_ev)

        return events

    # ----------------------------------------------------------------
    # STAGE 1: FIRE visual detection + feature extraction
    # ----------------------------------------------------------------
    def _fire_pipeline(self, zone_name, roi_mask, bbox, hsv, fire_mask,
                       roi_area, fire_px_thresh, min_contour,
                       object_bboxes=None):
        """Fire: STAGE 1 (detection) → STAGE 2 (persistence)
        → STAGE 3 (growth) → STAGE 4 (event)."""
        fire_in_roi = cv2.bitwise_and(fire_mask, roi_mask)
        fire_pixels = self._count_significant_contours(
            fire_in_roi, min_area=min_contour)

        # STAGE 1 — Visual features (kênh MÀU):
        flicker_score = self._compute_flicker(zone_name, fire_in_roi)
        contrast_thresh = (config.SMOKE_FIRE_GRID_CONTRAST_THRESH
                           if fire_px_thresh != config.SMOKE_FIRE_FIRE_PIXEL_THRESH
                           else config.SMOKE_FIRE_CONTRAST_THRESH)
        fire_contrast = self._compute_fire_contrast(
            fire_in_roi, roi_mask, hsv, min_area=min_contour)
        flicker_freq = self._compute_flicker_frequency(
            zone_name, fire_pixels > fire_px_thresh)
        edge_irregularity = self._compute_fire_edge_irregularity(
            fire_in_roi, min_area=min_contour)
        hue_spread = self._compute_fire_hue_spread(
            fire_in_roi, hsv, min_area=min_contour)
        core_ratio = self._compute_fire_bright_core(
            fire_in_roi, hsv, min_area=min_contour)
        fire_class_score = self._classify_fire(
            fire_contrast, flicker_score, flicker_freq,
            edge_irregularity, hue_spread, core_ratio,
        )

        # --- PHÂN LOẠI LỬA (bản chất lửa từ MÀU) ---
        edge_ok = edge_irregularity > config.SMOKE_FIRE_FIRE_EDGE_IRREGULARITY_THRESH
        flicker_ok = (flicker_score > config.SMOKE_FIRE_FLICKER_THRESH
                      or flicker_freq > config.SMOKE_FIRE_FLICKER_FREQ_THRESH)
        hue_ok = hue_spread > config.SMOKE_FIRE_FIRE_HUE_CIRC_VAR_THRESH
        core_ok = core_ratio > config.SMOKE_FIRE_FIRE_BRIGHT_CORE_RATIO_THRESH
        fire_nature = hue_ok or core_ok

        # --- LOẠI VẬT THỂ MÀU ĐỎ (áo quần người) ---
        # Vùng "lửa" nằm chủ yếu TRONG bbox người = áo đỏ/cam, không phải lửa.
        person_overlap = self._fire_inside_objects_ratio(
            fire_in_roi, object_bboxes)
        is_object_clothing = (
            object_bboxes
            and person_overlap > config.SMOKE_FIRE_MAX_PERSON_OVERLAP)

        # STAGE 3 — Region growth / motion:
        area_fluct = self._compute_fire_area_fluct(zone_name, fire_pixels)

        is_grid_cell = (fire_px_thresh != config.SMOKE_FIRE_FIRE_PIXEL_THRESH)
        area_fluct_thresh = (config.SMOKE_FIRE_GRID_AREA_FLUCT_THRESH
                             if is_grid_cell
                             else config.SMOKE_FIRE_FIRE_AREA_FLUCT_THRESH)
        jagged_thresh = (config.SMOKE_FIRE_GRID_JAGGED_THRESH
                         if is_grid_cell
                         else config.SMOKE_FIRE_FIRE_JAGGED_THRESH)

        # 3 đường xác nhận lửa:
        dynamic_path = fire_nature and (flicker_ok or edge_ok)
        static_path = (core_ok and hue_ok)
        # Đường "lửa đồng màu cháy âm ỉ": bắt buộc TẦN SỐ nhấp nháy thật
        # (bật/tắt giữa các frame). Vật thể di chuyển (xe đỏ) tuy có flicker_score
        # do cạnh thay đổi nhưng luôn present → flicker_freq ≈ 0 → bị loại.
        uniform_flicker_path = (
            flicker_freq > config.SMOKE_FIRE_FLICKER_FREQ_THRESH
            and (area_fluct > area_fluct_thresh
                 or edge_irregularity > jagged_thresh)
        )

        # LỬA THẬT phải có bản chất lửa (đa sắc HOẶC lõi sáng): vật đỏ/cam thuần
        # không thỏa dù có nhấp nháy cạnh (màu đồng nhất tuyệt đối, không lõi).
        # 3 đường nêu trên tự đòi fire_nature (dynamic/static), chỉ đường
        # uniform_flicker_path (cháy âm ỉ) cần fire_nature bổ sung.
        is_fire_signal = (
            fire_pixels > fire_px_thresh
            and fire_contrast > contrast_thresh
            and not is_object_clothing
            and (dynamic_path or static_path
                 or (uniform_flicker_path and fire_nature))
        )

        # STAGE 2 — Temporal persistence (cap để decay nhanh khi hết tín hiệu)
        if is_fire_signal:
            self.fire_persistence_map[zone_name] = min(
                config.SMOKE_FIRE_FIRE_PERSIST_THRESH + 2,
                self.fire_persistence_map.get(zone_name, 0) + 1,
            )
        else:
            self.fire_persistence_map[zone_name] = max(
                0, self.fire_persistence_map.get(zone_name, 0) - 1
            )

        # STAGE 3 — Region growth: diện tích tăng dần → tăng confidence
        hist = self.fire_area_history.setdefault(zone_name, [])
        hist.append(fire_pixels)
        if len(hist) > config.SMOKE_FIRE_GROWTH_WINDOW:
            hist.pop(0)
        growth = 0.0
        if (len(hist) >= 3
                and hist[-1] > hist[-2] > hist[-3]
                and hist[-1] > fire_px_thresh):
            growth = 0.08

        # STAGE 4 — FIRE EVENT
        if (self.fire_persistence_map.get(zone_name, 0) >= config.SMOKE_FIRE_FIRE_PERSIST_THRESH
                and self._alert_ready(zone_name)):
            self._mark_alert(zone_name)
            conf = min(0.98, 0.90 + growth
                       + min(fire_class_score, 1.0) * 0.06
                       + min(fire_contrast / 300.0, 0.03))
            fire_ev = {
                "event_type": "FIRE_DETECTED",
                "zone_name": zone_name,
                "confidence": conf,
                "description": f"Phát hiện đám cháy tại ô {zone_name}",
            }
            if bbox is not None:
                fire_ev["bbox"] = bbox
            return fire_ev
        return None

    # ----------------------------------------------------------------
    # SMALL FIRE — điểm cháy nhỏ / lửa bị che khuất MỘT PHẦN
    # ----------------------------------------------------------------
    def _small_fire_pipeline(self, zone_name, roi_mask, bbox, hsv, fire_mask,
                             px_thresh, min_contour, object_bboxes=None):
        """Lửa NHỎ (vài chục pixel) hoặc lửa chỉ lộ ra ít pixel sau vật cản.

        Ngọn lửa nhỏ vẫn giữ bản chất lửa:
          - NHẤP NHÁY liên tục (tín hiệu tin cậy nhất với lửa nhỏ).
          - Đa sắc vàng→cam→đỏ (hue spread) hoặc lõi sáng trắng/vàng nhạt.
        KHÔNG đòi hỏi edge_irregularity (blob nhỏ không đáng tin) hay contrast
        cao (ánh lửa hắt vào vật cản làm nền sáng → contrast giảm). Ngưỡng
        pixel/contour thấp hơn pipeline fire chính để bắt được lửa nhỏ.
        """
        fire_in_roi = cv2.bitwise_and(fire_mask, roi_mask)
        fire_pixels = self._count_significant_contours(
            fire_in_roi, min_area=min_contour)

        flicker_score = self._compute_flicker("small|" + zone_name, fire_in_roi)
        flicker_freq = self._compute_flicker_frequency(
            "small|" + zone_name, fire_pixels > px_thresh)
        hue_spread = self._compute_fire_hue_spread(
            fire_in_roi, hsv, min_area=min_contour)
        core_ratio = self._compute_fire_bright_core(
            fire_in_roi, hsv, min_area=min_contour)

        flicker_ok = (flicker_score > config.SMOKE_FIRE_FLICKER_THRESH
                      or flicker_freq > config.SMOKE_FIRE_FLICKER_FREQ_THRESH)
        fire_nature_ok = (hue_spread > config.SMOKE_FIRE_SMALL_FIRE_HUE_VAR_THRESH
                          or core_ratio > config.SMOKE_FIRE_SMALL_FIRE_CORE_RATIO_THRESH)

        # Vùng "lửa" nằm chủ yếu trong bbox NGƯỜI = áo quần đỏ/cam → loại.
        person_overlap = self._fire_inside_objects_ratio(
            fire_in_roi, object_bboxes)
        is_object_clothing = (
            object_bboxes
            and person_overlap > config.SMOKE_FIRE_MAX_PERSON_OVERLAP)

        is_small_fire = (fire_pixels > px_thresh
                         and flicker_ok and fire_nature_ok
                         and not is_object_clothing)

        # Ghi tín hiệu vào frame hiện tại (cho bộ gom lửa di động) — chỉ cho ô grid
        if is_small_fire and bbox is not None and "|" in zone_name:
            base, cell = zone_name.split("|", 1)
            self._frame_cell_signals.setdefault(base, set()).add(cell)
            self._frame_cell_bboxes.setdefault(base, []).append(bbox)

        # STAGE 2 — Temporal persistence (key riêng để không trộn với lửa chính)
        # Decay chuẩn (giống lửa chính): khi mất tín hiệu → counter giảm dần về 0.
        # Trước đây dùng hold-zone (giữ counter khi còn >= 50% ngưỡng) khiến counter
        # nằm mãi trên ngưỡng → báo lặp lại mọi frame. Cooldown + decay chuẩn sẽ chặn.
        key = "small|" + zone_name
        if is_small_fire:
            self.fire_persistence_map[key] = min(
                config.SMOKE_FIRE_SMALL_FIRE_PERSIST_THRESH + 2,
                self.fire_persistence_map.get(key, 0) + 1,
            )
        else:
            self.fire_persistence_map[key] = max(
                0, self.fire_persistence_map.get(key, 0) - 1
            )

        # STAGE 4 — FIRE EVENT
        if (self.fire_persistence_map.get(key, 0)
                >= config.SMOKE_FIRE_SMALL_FIRE_PERSIST_THRESH
                and self._alert_ready(key)):
            self._mark_alert(key)
            fire_ev = {
                "event_type": "FIRE_DETECTED",
                "zone_name": zone_name,
                "confidence": min(0.97, 0.90
                                  + min(flicker_freq, 1.0) * 0.06
                                  + min(hue_spread, 1.0) * 0.03),
                "description": f"Phát hiện điểm cháy nhỏ tại ô {zone_name}",
            }
            if bbox is not None:
                fire_ev["bbox"] = bbox
            return fire_ev
        return None

    # ----------------------------------------------------------------
    # MOVING SMALL FIRE — nguồn sáng nhỏ KHÔNG CỐ ĐỊNH, di chuyển giữa các ô
    # ----------------------------------------------------------------
    def _aggregate_moving_fire(self, base_name, cell_keys, cell_bboxes,
                               object_bboxes=None):
        """Gom tín hiệu lửa nhỏ ở mức ROI để bắt điểm lửa di động.

        Điểm lửa nhỏ (tàn lửa, ngọn đuốc, ngọn lửa nhỏ bị gió lay...) di chuyển
        giữa các ô grid: mỗi ô chỉ thấy nó 1-2 frame → persistence từng ô không
        đủ. Bộ gom này tích lũy tín hiệu ở mức ROI và chỉ báo khi:
          - Có tín hiệu lửa nhỏ liên tục (counter theo persistence).
          - Lửa xuất hiện ở >= 2 ô KHÁC NHAU trong cửa sổ → chứng tỏ DI ĐỘNG
            (không phải 1 điểm cố định — điểm cố định do pipeline từng ô xử lý).
        Vùng tín hiệu nằm trong bbox NGƯỜI (áo đỏ/cam di chuyển) bị loại.
        """
        key = "move|" + base_name

        # Loại bỏ tín hiệu nằm chủ yếu trong bbox người (áo đỏ/cam) trước khi
        # đếm: hợp nhất tất cả cell bbox của frame này thành 1 mask, nếu phần
        # lớn diện tích đó nằm trong người → đó là người mặc áo đỏ, không phải
        # điểm lửa di động → coi như không có tín hiệu (counter sẽ decay).
        if cell_bboxes and object_bboxes:
            frame_h = max(bb[3] for bb in cell_bboxes)
            frame_w = max(bb[2] for bb in cell_bboxes)
            cell_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            for bb in cell_bboxes:
                x1, y1, x2, y2 = [int(v) for v in bb]
                cv2.rectangle(cell_mask, (x1, y1), (x2, y2), 255, -1)
            r = self._fire_inside_objects_ratio(cell_mask, object_bboxes)
            if r > config.SMOKE_FIRE_MAX_PERSON_OVERLAP:
                cell_keys = set()
                cell_bboxes = []

        if cell_keys:
            self.moving_fire_counter[key] = min(
                config.SMOKE_FIRE_SMALL_FIRE_PERSIST_THRESH + 2,
                self.moving_fire_counter.get(key, 0) + 1,
            )
        else:
            self.moving_fire_counter[key] = max(
                0, self.moving_fire_counter.get(key, 0) - 1
            )
            return None

        hist = self.moving_fire_cells.setdefault(key, [])
        hist.append((set(cell_keys), list(cell_bboxes)))
        if len(hist) > config.SMOKE_FIRE_SMALL_FIRE_PERSIST_THRESH:
            hist.pop(0)

        distinct_cells = set()
        all_bboxes = []
        for ck, bb in hist:
            distinct_cells.update(ck)
            all_bboxes.extend(bb)

        # Chỉ báo khi có DI ĐỘNG thật (>= 2 ô khác nhau) + tín hiệu kéo dài
        if (len(distinct_cells) < 2
                or self.moving_fire_counter[key]
                < config.SMOKE_FIRE_SMALL_FIRE_PERSIST_THRESH
                or not self._alert_ready(key)):
            return None
        self._mark_alert(key)

        x1 = min(b[0] for b in all_bboxes)
        y1 = min(b[1] for b in all_bboxes)
        x2 = max(b[2] for b in all_bboxes)
        y2 = max(b[3] for b in all_bboxes)

        return {
            "event_type": "FIRE_DETECTED",
            "zone_name": base_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": min(0.97, 0.93 + len(distinct_cells) * 0.01),
            "description":
                f"Phát hiện điểm lửa nhỏ di động (nguồn sáng không cố định) "
                f"tại khu vực {base_name}",
        }

    # ----------------------------------------------------------------
    # FIRE GLOW — lửa bị che khuất hoàn toàn (tường, container, bờ kè...)
    # ----------------------------------------------------------------
    def _fire_glow_pipeline(self, zone_name, roi_mask, bbox, hsv, glow_mask,
                            px_thresh, min_contour, object_bboxes=None):
        """Quầng sáng ấm hắt ra vật cản — bản thân ngọn lửa không hiện rõ.

        Lửa ẩn sau tường vẫn:
          - Chiếu sáng bề mặt xung quanh bằng ánh cam mờ (glow mask).
          - NHẤP NHÁY theo ngọn lửa bên trong (flicker / flicker_freq).
          - Tạo vùng ấm đa sắc (hue spread).
        Duy trì qua nhiều frame (persistence) trước khi báo → loại vật tĩnh
        màu ấm (tường nắng, đèn) không nhấp nháy.
        """
        glow_in_roi = cv2.bitwise_and(glow_mask, roi_mask)
        glow_pixels = self._count_significant_contours(
            glow_in_roi, min_area=min_contour)

        flicker_score = self._compute_flicker("glow|" + zone_name, glow_in_roi)
        flicker_freq = self._compute_flicker_frequency(
            "glow|" + zone_name, glow_pixels > px_thresh)
        # Đa sắc ấm — lửa thật luôn trải nhiều sắc độ (đỏ→cam→vàng); áo quần
        # đỏ/cam đồng nhất một sắc độ → hue variance ≈ 0. Gate chặn vật đỏ.
        hue_spread = self._compute_fire_hue_spread(
            glow_in_roi, hsv, min_area=min_contour)
        hue_ok = hue_spread > config.SMOKE_FIRE_GLOW_HUE_VAR_THRESH

        # Vùng ấm nằm chủ yếu trong bbox NGƯỜI = áo quần đỏ/cam → loại.
        person_overlap = self._fire_inside_objects_ratio(
            glow_in_roi, object_bboxes)
        is_object_clothing = (
            object_bboxes
            and person_overlap > config.SMOKE_FIRE_MAX_PERSON_OVERLAP)

        # Gate: vùng ấm hiện diện + NHẤP NHÁY (lửa ẩn làm quầng sáng pulse)
        # + đa sắc ấm (bản chất lửa) + kéo dài (persistence). Vật tĩnh màu ấm
        # không nhấp nháy / vật đỏ không đa sắc → bị loại.
        is_glow_signal = (
            glow_pixels > px_thresh
            and (flicker_score > config.SMOKE_FIRE_GLOW_FLICKER_THRESH
                 or flicker_freq > config.SMOKE_FIRE_FLICKER_FREQ_THRESH)
            and hue_ok
            and not is_object_clothing
        )

        # STAGE 2 — Temporal persistence
        # Decay chuẩn: khi quầng không còn nhấp nháy/đa sắc/đủ pixel → counter giảm
        # về 0. Trước đây dùng hold-zone giữ counter mãi trên ngưỡng → 1 bề mặt ấm
        # tĩnh (tường nắng, xe cam) báo lặp lại vô tận. Cooldown chặn báo lặp.
        key = "glow|" + zone_name
        if is_glow_signal:
            self.fire_persistence_map[key] = min(
                config.SMOKE_FIRE_GLOW_PERSIST_THRESH + 2,
                self.fire_persistence_map.get(key, 0) + 1,
            )
        else:
            self.fire_persistence_map[key] = max(
                0, self.fire_persistence_map.get(key, 0) - 1
            )

        # STAGE 4 — FIRE EVENT
        if (self.fire_persistence_map.get(key, 0)
                >= config.SMOKE_FIRE_GLOW_PERSIST_THRESH
                and self._alert_ready(key)):
            self._mark_alert(key)
            glow_ev = {
                "event_type": "FIRE_DETECTED",
                "zone_name": zone_name,
                "confidence": min(0.96, 0.90
                                  + min(flicker_freq, 1.0) * 0.05
                                  + min(hue_spread, 1.0) * 0.04),
                "description":
                    f"Phát hiện quầng sáng cháy (lửa có thể bị che khuất) tại ô {zone_name}",
            }
            if bbox is not None:
                glow_ev["bbox"] = bbox
            return glow_ev
        return None

    # ----------------------------------------------------------------
    # STAGE 1: SMOKE visual detection + feature extraction
    # ----------------------------------------------------------------
    def _smoke_pipeline(self, zone_name, roi_mask, bbox, hsv, smoke_mask,
                        roi_area, smoke_px_thresh, min_contour):
        """Smoke: STAGE 1 (detection) → STAGE 2 (persistence)
        → STAGE 3 (growth/spread) → STAGE 4 (event)."""
        smoke_in_roi = cv2.bitwise_and(smoke_mask, roi_mask)
        smoke_pixels = self._count_significant_contours(
            smoke_in_roi, min_area=min_contour)

        # STAGE 1 — Visual features. Grayscale chỉ phụ trợ cho khói
        # (softness / frame diff), không grayscale toàn pipeline.
        gray = self._gray_from_hsv(hsv)

        smoke_change = self._compute_smoke_change(zone_name, gray, roi_mask)
        smoke_softness = self._compute_smoke_softness(
            smoke_in_roi, gray, min_area=min_contour)
        shape_change, spread = self._compute_smoke_shape_change(
            zone_name, smoke_in_roi)

        smoke_ratio = smoke_pixels / max(roi_area, 1)

        # STAGE 1 signal = đủ pixel + mờ (dạng mây) + đổi hình/lan tỏa + frame diff
        is_smoke_signal = (
            smoke_pixels > smoke_px_thresh
            and smoke_softness > config.SMOKE_FIRE_SMOKE_SOFTNESS_THRESH
            and (shape_change > config.SMOKE_FIRE_SMOKE_SHAPE_CHANGE_THRESH
                 or spread > 0.05)
            and (smoke_change > config.SMOKE_FIRE_SMOKE_CHANGE_THRESH
                 or smoke_ratio > 0.15)
        )

        # STAGE 2 — Temporal persistence
        if is_smoke_signal:
            self.smoke_persistence_map[zone_name] = min(
                config.SMOKE_FIRE_SMOKE_PERSIST_THRESH + 3,
                self.smoke_persistence_map.get(zone_name, 0) + 1,
            )
        else:
            self.smoke_persistence_map[zone_name] = max(
                0, self.smoke_persistence_map.get(zone_name, 0) - 1
            )

        # STAGE 4 — SMOKE EVENT (STAGE 3: shape_change/spread đã tính ở trên)
        if (self.smoke_persistence_map.get(zone_name, 0)
                >= config.SMOKE_FIRE_SMOKE_PERSIST_THRESH
                and self._alert_ready("smoke|" + zone_name)):
            self._mark_alert("smoke|" + zone_name)
            smoke_ev = {
                "event_type": "SMOKE_DETECTED",
                "zone_name": zone_name,
                "confidence": min(0.98, 0.90 + smoke_change * 2
                                  + smoke_softness * 0.2 + spread * 0.5),
                "description": f"Phát hiện khói bất thường tại ô {zone_name}",
            }
            if bbox is not None:
                smoke_ev["bbox"] = bbox
            return smoke_ev
        return None

    def _gray_from_hsv(self, hsv):
        """Trích kênh V (giá trị/độ sáng) từ HSV làm grayscale phụ trợ.

        Khói mờ được nhận diện qua độ sáng trung bình; tránh convert toàn
        bộ pipeline về grayscale (màu vẫn giữ cho fire detection).
        """
        return hsv[:, :, 2].copy()

    # ----------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------

    def _count_significant_contours(self, mask, min_area=None):
        """Chỉ đếm pixel từ contour có diện tích >= ngưỡng (loại bỏ nhiễu nhỏ)."""
        if min_area is None:
            min_area = config.SMOKE_FIRE_MIN_CONTOUR_AREA
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
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

        if total < 20:
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

    def _compute_fire_contrast(self, fire_in_roi, roi_mask, hsv, min_area=None):
        """Độ sáng tương phản của lửa so với nền tối xung quanh.

        Lửa thật rất sáng (V cao) nổi bật trên nền tối. Vật thể ấm sáng trên nền
        sáng (đường, trời sáng) có độ tương phản thấp → bị loại.
        """
        if min_area is None:
            min_area = config.SMOKE_FIRE_MIN_CONTOUR_AREA
        fire_px = cv2.countNonZero(fire_in_roi)
        if fire_px < min_area:
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

    def _compute_fire_edge_irregularity(self, fire_in_roi, min_area=None):
        """Cạnh sắc không đều của vùng lửa (edge irregularity / jaggedness).

        Dùng compactness = 4*pi*A / P². Hình tròn mượt (đèn, nguồn sáng) → ≈1.
        Lửa thật có đường viền răng cưa, xù xì → tỷ lệ nghịch (irregularity > 1).
        Lấy giá trị cao nhất trong các contour đủ lớn (ngọn lửa chính).
        """
        if min_area is None:
            min_area = config.SMOKE_FIRE_MIN_CONTOUR_AREA
        contours, _ = cv2.findContours(fire_in_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        irregularity = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            compactness = (4.0 * np.pi * area) / (perimeter * perimeter)
            if compactness > 0:
                irregularity = max(irregularity, 1.0 / compactness)
        return irregularity

    def _classify_fire(self, fire_contrast, flicker_score, flicker_freq,
                       edge_irregularity, hue_spread, core_ratio):
        """Phân loại đối tượng: kết hợp đặc trưng thành điểm LỬA (0..1).

        So sánh với cơ sở mẫu lửa thật: phải ĐỒNG THỜI có màu sáng nổi bật trên
        nền tối (contrast), nhấp nháy (flicker) và cạnh sắc không đều. Vật thể
        màu tương tự (đèn, mây, vật đỏ tĩnh) chỉ đạt 1-2 đặc trưng → điểm thấp.
        """
        # Chuẩn hoá từng đặc trưng về [0..1]
        norm_contrast = min(max(fire_contrast / 150.0, 0.0), 1.0)
        norm_flicker = min(max(flicker_score / 0.4, 0.0), 1.0)
        norm_freq = min(max(flicker_freq / 0.6, 0.0), 1.0)
        norm_edge = min(max((edge_irregularity - 1.0) / 1.5, 0.0), 1.0)
        norm_hue = min(max(hue_spread / 0.25, 0.0), 1.0)
        norm_core = min(max(core_ratio / 0.25, 0.0), 1.0)

        # Lửa thật: contrast CAO (bắt buộc) + cạnh sắc răng cưa + nhấp nháy
        # + BẢN CHẤT LỬA (đa sắc hoặc lõi sáng). Vật đỏ đồng màu tĩnh có
        # norm_hue ≈ 0 và norm_core ≈ 0 → điểm thấp dù edge/contrast cao.
        score = (
            norm_contrast * 0.20
            + norm_flicker * 0.20
            + norm_freq * 0.15
            + norm_edge * 0.15
            + norm_hue * 0.15
            + norm_core * 0.15
        )
        return score

    def _compute_fire_hue_spread(self, fire_in_roi, hsv, min_area=None):
        """Độ phân tán màu (hue circular variance) trong vùng lửa.

        Lửa thật chứa nhiều sắc độ: lõi vàng nhạt → cam → đỏ (hue trải rộng).
        Vật đỏ đồng nhất (áo, xe, biển báo) có hue tập trung 1 sắc → variance
        ≈ 0. Circular variance tránh lỗi wrap-around của hue (đỏ = 0/180).
        """
        if min_area is None:
            min_area = config.SMOKE_FIRE_MIN_CONTOUR_AREA
        mask = fire_in_roi > 0
        if np.count_nonzero(mask) < min_area:
            return 0.0

        hue = hsv[:, :, 0].astype(np.float32)[mask]
        sat = hsv[:, :, 1].astype(np.float32)[mask]

        # Chỉ xét pixel có màu rõ (S đủ cao) — bỏ vùng trắng/xám nhiễu
        color_mask = sat > 60
        if np.count_nonzero(color_mask) < 10:
            return 0.0
        hue = hue[color_mask]

        # Circular variance: hue/180 * 2π, đỏ (0/180) trở thành cùng góc
        theta = (hue / 180.0) * (2.0 * np.pi)
        cos_sum = np.mean(np.cos(theta))
        sin_sum = np.mean(np.sin(theta))
        r = float(np.hypot(cos_sum, sin_sum))
        return max(0.0, 1.0 - r)

    def _compute_fire_bright_core(self, fire_in_roi, hsv, min_area=None):
        """Tỷ lệ pixel lõi sáng (trắng/vàng nhạt) TRONG/NHẠT vùng lửa.

        Lửa thật có lõi rất sáng (V cao) và ít bão hòa (S thấp) nằm GIỮA ngọn
        lửa. Lõi trắng/vàng nhạt thường bị loại khỏi fire mask (mask yêu cầu
        S >= 90) nên phải NỞ RỘNG vùng lửa (dilate) trước khi đo. Vật đỏ/cam
        thuần (S cao khắp nơi — áo đỏ, xe đỏ, đèn) không có pixel sáng bão hòa
        thấp nào → ratio ≈ 0. Đo thực tế: lửa ~0.045, vật đỏ ~0.000.
        """
        if min_area is None:
            min_area = config.SMOKE_FIRE_MIN_CONTOUR_AREA
        if cv2.countNonZero(fire_in_roi) < min_area:
            return 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(fire_in_roi, kernel)

        v = hsv[:, :, 2].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        core = ((dilated > 0)
                & (v > config.SMOKE_FIRE_FIRE_CORE_V_THRESH)
                & (s < config.SMOKE_FIRE_FIRE_CORE_S_MAX))
        total = int(np.count_nonzero(dilated > 0))
        if total == 0:
            return 0.0
        return int(np.count_nonzero(core)) / max(total, 1)

    def _fire_inside_objects_ratio(self, fire_mask_roi, object_bboxes):
        """Tỷ lệ pixel "lửa" nằm BÊN TRONG bbox của object đã detect (người...).

        Người mặc áo đỏ/cam → vùng màu lửa nằm gần như trọn trong bbox người
        → ratio cao → coi là vật thể, không phải lửa (loại bỏ false positive).
        Lửa thật không nằm trong bbox người → ratio thấp → giữ nguyên.
        """
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

    def _compute_fire_area_fluct(self, zone_name, current_pixels):
        """Hệ số biến thiên (CV = std/mean) của diện tích lửa trong cửa sổ.

        Lửa co/giãn liên tục → diện tích dao động mạnh (CV cao). Vật đỏ di
        chuyển (xe, người) giữ nguyên kích thước → CV thấp. Phân biệt lửa đồng
        màu cháy âm ỉ với vật đỏ động mà không cần lõi/hue.
        """
        hist = list(self.fire_area_history.get(zone_name, []))
        if not hist:
            return 0.0
        hist.append(current_pixels)
        if len(hist) > config.SMOKE_FIRE_GROWTH_WINDOW:
            hist = hist[-config.SMOKE_FIRE_GROWTH_WINDOW:]

        arr = np.array(hist, dtype=np.float32)
        mean = float(arr.mean())
        if mean < 1e-5:
            return 0.0
        return float(arr.std()) / mean

    def _compute_smoke_softness(self, smoke_in_roi, gray_frame, min_area=None):
        """Độ trong suốt/mờ nội tại của vùng khói.

        Khói là mây mờ lan tỏa: nội tại gradient thấp, không có cạnh sắc.
        Đối tượng đặc (xe, tường, người) có cạnh/chi tiết → gradient cao.
        """
        if min_area is None:
            min_area = config.SMOKE_FIRE_MIN_CONTOUR_AREA
        if cv2.countNonZero(smoke_in_roi) < min_area:
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
