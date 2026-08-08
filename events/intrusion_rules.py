import numpy as np
import cv2
import config

class IntrusionRules:
    """Rules cho xâm nhập vùng cấm: nhiều spatial ROI (zone) + dwell time riêng từng vùng."""

    def check(self, camera_id, obj):
        """Cập nhật dwell time của obj cho từng zone. Trả về danh sách zone bị xâm nhập:
        [{"zone_name": ..., "dwell_frames": ...}, ...]"""
        zones = config.get_roi_zones(camera_id)
        if not zones:
            return []

        curr_center = obj.center_history[-1]
        intruded = []

        for zone in zones:
            zone_name = zone["name"]
            roi_pts = np.array(zone["polygon"], np.int32)
            is_inside = cv2.pointPolygonTest(
                roi_pts, (float(curr_center[0]), float(curr_center[1])), False
            ) >= 0

            obj.tick_dwell(zone_name, is_inside)

            if obj.dwell_times.get(zone_name, 0) >= config.INTRUSION_DWELL_FRAMES:
                intruded.append({
                    "zone_name": zone_name,
                    "dwell_frames": obj.dwell_times.get(zone_name, 0),
                })

        return intruded