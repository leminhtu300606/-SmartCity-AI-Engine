import cv2
import numpy as np
import config


class SmokeFireRules:
    """Visual pattern analysis giữ nguyên màu Color Space + Temporal Persistence + Region Growth."""

    def __init__(self):
        self.fire_persistence_map = {}   # zone -> frame_count
        self.smoke_persistence_map = {}
        self.fire_area_history = {}      # zone -> [fire_pixels, ...] để tính region growth

    def analyze_frame(self, frame_bgr, roi_polygons):
        events = []
        if frame_bgr is None:
            return events

        # GIỮ NGUYÊN MÀU: Chuyển HSV để bắt màu Lửa & Khói
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Fire Color Mask (Vàng, Cam, Đỏ tươi)
        lower_fire = np.array([0, 120, 180], dtype=np.uint8)
        upper_fire = np.array([35, 255, 255], dtype=np.uint8)
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

        # Smoke Color Mask (Xám, Trắng, Độ bão hòa thấp)
        lower_smoke = np.array([0, 0, 100], dtype=np.uint8)
        upper_smoke = np.array([180, 30, 230], dtype=np.uint8)
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

        for roi in roi_polygons:
            if roi.get("event_type") != config.EVENT_TYPE_SMOKE_FIRE:
                continue

            zone_name = roi["name"]
            poly = np.array(roi["polygon"], np.int32)
            
            # Mask hóa ROI
            roi_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [poly], 255)

            fire_pixels = cv2.countNonZero(cv2.bitwise_and(fire_mask, roi_mask))
            smoke_pixels = cv2.countNonZero(cv2.bitwise_and(smoke_mask, roi_mask))

            # Temporal Persistence Increment
            if fire_pixels > 150:
                self.fire_persistence_map[zone_name] = self.fire_persistence_map.get(zone_name, 0) + 1
            else:
                self.fire_persistence_map[zone_name] = max(0, self.fire_persistence_map.get(zone_name, 0) - 1)

            if smoke_pixels > 300:
                self.smoke_persistence_map[zone_name] = self.smoke_persistence_map.get(zone_name, 0) + 1
            else:
                self.smoke_persistence_map[zone_name] = max(0, self.smoke_persistence_map.get(zone_name, 0) - 1)

            # Region growth: diện tích lửa tăng dần qua các frame -> tăng confidence
            hist = self.fire_area_history.setdefault(zone_name, [])
            hist.append(fire_pixels)
            if len(hist) > 4:
                hist.pop(0)
            growth = 0.1 if len(hist) >= 3 and hist[-1] > hist[-2] and hist[-2] >= hist[-3] else 0.0

            # Fire / Smoke Confirm
            if self.fire_persistence_map[zone_name] >= 5:
                events.append({
                    "event_type": "FIRE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": min(0.98, 0.88 + growth),
                    "description": "Phát hiện đám cháy trong vùng quan sát"
                })

            if self.smoke_persistence_map[zone_name] >= 8:
                events.append({
                    "event_type": "SMOKE_DETECTED",
                    "zone_name": zone_name,
                    "confidence": 0.82,
                    "description": "Phát hiện khói bất thường"
                })

        return events