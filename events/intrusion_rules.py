import cv2
import numpy as np
import config


class IntrusionRules:
    """Spatial Rules: Person enters Restricted ROI -> Movement + Dwell Time."""

    def check_intrusion(self, obj, roi_zones):
        events = []
        if obj.cls_id != 0:  # Chỉ xét Person (Class 0)
            return events

        # Không tính dwell trên frame dự đoán (vị trí chưa phải thật) -> tránh alert sớm
        if obj.last_update_predicted:
            return events

        curr_center = obj.center_history[-1]

        for zone in roi_zones:
            if zone["event_type"] != config.EVENT_TYPE_INTRUSION:
                continue

            zone_name = zone["name"]
            polygon = np.array(zone["polygon"], np.int32)
            
            # Point in Polygon Test
            is_inside = cv2.pointPolygonTest(polygon, (float(curr_center[0]), float(curr_center[1])), False) >= 0

            # Tick dwell time
            obj.tick_dwell(zone_name, is_inside)

            # Check Movement trong vùng
            speeds = [np.linalg.norm(v) for v in list(obj.velocity_history)[-5:]]
            avg_speed = np.mean(speeds) if len(speeds) > 0 else 0.0

            if is_inside and obj.dwell_times[zone_name] >= config.INTRUSION_DWELL_FRAMES and avg_speed > 0.5:
                events.append({
                    "event_type": "RESTRICTED_INTRUSION",
                    "track_ids": [obj.track_id],
                    "zone_name": zone_name,
                    "confidence": 0.95,
                    "description": f"Phát hiện xâm nhập khu vực cấm {zone_name}"
                })

        return events