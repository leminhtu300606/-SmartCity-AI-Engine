import cv2
import numpy as np
import config
from events.scores import fire_score, smoke_score


class SmokeFireRules:
    """Phát hiện LỬA & KHÓI — EMIT SCORED CANDIDATE (Rule 6D/6E).

    Rule 6D: "màu đỏ/cam/sáng" một mình KHÔNG phải FIRE.
      FireScore = 0.35*fire_model + 0.20*spatial_consistency
                 + 0.20*temporal_persistence + 0.15*fire_motion
                 + 0.10*smoke_corroboration ; >= 0.85 mới CONFIRMED.
    Rule 6E: "gray blur" KHÔNG phải khói. Khói phải phát triển / lan / biến dạng
      (spatial expansion) chứ không phải vùng xám tĩnh.
    """

    def __init__(self):
        self.fire_persist_window = {}    # zone -> list[0/1]
        self.smoke_persist_window = {}   # zone -> list[0/1]
        self.prev_fire_mask = {}         # zone -> prev fire mask (flicker)
        self.prev_smoke_area = {}        # zone -> area khói frame trước (expansion)
        self._frame_count = 0

    # ================================================================
    # ORCHESTRATOR — Level 2/3: phân tích frame (chạy ở FIRE_FPS cadence)
    # ================================================================
    def analyze_frame(self, frame_bgr, roi_polygons, object_bboxes=None):
        """Trả list CANDIDATE: FIRE_DETECTED / SMOKE_DETECTED (chưa confirm).

        Candidate được phát MỌI frame đủ score -> EventConfirmTracker mới
        tích luỹ temporal để đưa lên CONFIRMED (Rule 7/14). Việc chống
        spam alert lặp đã do handler (main) dedup theo key.
        """
        events = []
        if frame_bgr is None:
            return events

        self._frame_count += 1
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        fire_mask = self._detect_fire_mask(hsv)
        smoke_mask = self._detect_smoke_mask(hsv)
        smoke_area = cv2.countNonZero(smoke_mask)

        for roi in roi_polygons:
            if roi.get("event_type") != config.EVENT_TYPE_SMOKE_FIRE:
                continue

            zone_name = roi["name"]
            poly = np.array(roi["polygon"], np.int32)
            roi_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [poly], 255)

            fire_in_roi = cv2.bitwise_and(fire_mask, roi_mask)
            smoke_in_roi = cv2.bitwise_and(smoke_mask, roi_mask)

            # ----- FIRE components -----
            fire_pixels = self._count_significant_contours(fire_in_roi)
            hue_spread = self._compute_fire_hue_spread(fire_in_roi, hsv)
            flicker = self._compute_flicker(zone_name, fire_in_roi)
            person_overlap = self._fire_inside_objects_ratio(
                fire_in_roi, object_bboxes)
            clothing = (object_bboxes
                        and person_overlap > config.SMOKE_FIRE_MAX_PERSON_OVERLAP)

            fire_signal = 1.0 if (fire_pixels > config.SMOKE_FIRE_FIRE_PIXEL_THRESH
                                  and not clothing) else 0.0

            # smoke corroboration (0..1): khói gần vùng lửa
            smoke_score_v = self._corroboration_smoke(smoke_in_roi, fire_in_roi)

            persist_hits, persist_win = self._persist_run(
                "fire", zone_name, bool(fire_signal))

            comps, fire_sc = fire_score(
                fire_signal, hue_spread, flicker,
                persist_hits, persist_win, smoke_score_v)

            if fire_sc >= config.FIRE_CANDIDATE_THRESH:
                events.append({
                    "event_type": "FIRE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": round(fire_sc, 3),
                    "score_components": comps,
                    "description": "Phát hiện đám cháy tại khu vực %s (FireScore %.2f)" % (zone_name, fire_sc),
                    "bbox": self._fire_bbox(fire_in_roi, poly),
                })

            # ----- SMOKE components -----
            smoke_signal = 1.0 if (cv2.countNonZero(smoke_in_roi)
                                   >= config.SMOKE_FIRE_MIN_CONTOUR_AREA) else 0.0
            expansion = self._smoke_expansion(zone_name, smoke_in_roi, smoke_area)
            shape = self._smoke_shape_consistency(smoke_in_roi)
            s_persist, _ = self._persist_run("smoke", zone_name, bool(smoke_signal))

            s_comps, smoke_sc = smoke_score(
                smoke_signal, expansion, shape, s_persist, config.SMOKE_CONFIRM_FRAMES)

            # Rule 6E: "gray blur" KHÔNG phải khói. Khói PHẢI lan rộng
            # (expansion > 0). Vùng xám tĩnh → bỏ qua candidate.
            smoke_expanding = expansion >= getattr(
                config, "SMOKE_MIN_EXPANSION", 0.05)
            if smoke_sc >= config.SMOKE_CANDIDATE_THRESH and smoke_signal \
                    and smoke_expanding:
                events.append({
                    "event_type": config.EVENT_TYPE_SMOKE,
                    "zone_name": zone_name,
                    "confidence": round(smoke_sc, 3),
                    "score_components": s_comps,
                    "description": "Phát hiện khói tại khu vực %s (SmokeScore %.2f)" % (zone_name, smoke_sc),
                    "bbox": self._blob_bbox(smoke_in_roi, poly),
                })

        return events

    # ----------------------------------------------------------------
    # PERSISTENCE WINDOW helper (dùng riêng cho fire/smoke — temporal)
    # ----------------------------------------------------------------
    def _persist_run(self, kind, zone_name, hit):
        window_kind = self.fire_persist_window if kind == "fire" else self.smoke_persist_window
        window = window_kind.setdefault(zone_name, [])
        window.append(1 if hit else 0)
        cap = (config.SMOKE_FIRE_FIRE_PERSIST_THRESH + 4) if kind == "fire" \
            else (config.SMOKE_CONFIRM_FRAMES + 4)
        if len(window) > cap:
            window.pop(0)
        return sum(window), len(window)

    def _compute_flicker(self, zone_name, fire_mask_current):
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

    def _smoke_expansion(self, zone_name, smoke_mask, total_smoke_area):
        """Spatial expansion: khói PHẢI lan rộng (tăng area) — not static gray blob."""
        prev = self.prev_smoke_area.get(zone_name)
        self.prev_smoke_area[zone_name] = cv2.countNonZero(smoke_mask)
        if prev is None:
            return 0.0
        if prev <= 0:
            return 0.0
        grow = (cv2.countNonZero(smoke_mask) - prev) / float(prev)
        return max(0.0, min(1.0, grow * 2.0))

    def _smoke_shape_consistency(self, smoke_mask):
        """Shape consistency: vùng khói phải có shape blob ổn định, không rải rác nhỏ."""
        n = cv2.countNonZero(smoke_mask)
        if n < config.SMOKE_FIRE_MIN_CONTOUR_AREA:
            return 0.0
        contours, _ = cv2.findContours(
            smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        biggest = max(contours, key=cv2.contourArea)
        return min(1.0, cv2.contourArea(biggest) / float(max(n, 1)))

    def _corroboration_smoke(self, smoke_mask, fire_mask):
        """Smoke corroboration: khói nằm sát/trong vùng lửa -> tăng tin cậy lửa."""
        if cv2.countNonZero(fire_mask) == 0:
            return 0.0
        radius = 40
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        fire_dilated = cv2.dilate(fire_mask, kernel)
        near = cv2.bitwise_and(smoke_mask, fire_dilated)
        return min(1.0, cv2.countNonZero(near) / float(max(cv2.countNonZero(smoke_mask), 1)))

    # ----------------------------------------------------------------
    # HỖ TRỢ KHÓI — Rule 6E: khói ≠ vùng xám tĩnh
    # Màu xám/low-sat, brightness biến thiên, shape blob. Chỉ là SIGNAL,
    # confirmation phải qua SmokeScore + persist + expansion.
    # ----------------------------------------------------------------
    def _detect_smoke_mask(self, hsv):
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        sat_lo = (s < 70)
        val_lo = (v >= 80) & (v <= 245)
        mask = (sat_lo & val_lo).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    # ----------------------------------------------------------------
    # FIRE MASK (giữ nguyên logic cũ)
    # ----------------------------------------------------------------
    def _detect_fire_mask(self, hsv):
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
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= config.SMOKE_FIRE_MIN_CONTOUR_AREA:
                total += int(area)
        return total

    def _compute_fire_hue_spread(self, fire_in_roi, hsv):
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
        ys, xs = np.where(fire_in_roi > 0)
        if len(xs) == 0:
            if poly is None:
                return None
            return [int(poly[:, 0].min()), int(poly[:, 1].min()),
                    int(poly[:, 0].max()), int(poly[:, 1].max())]
        return [int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max())]

    @staticmethod
    def _blob_bbox(mask, poly=None):
        return SmokeFireRules._fire_bbox(mask, poly)