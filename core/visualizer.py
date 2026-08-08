import cv2
import numpy as np
import config

class VideoVisualizer:
    """Vẽ ROI, bounding box + track ID, alerts và FPS lên frame."""

    def draw(self, frame, camera_id, memory_manager, active_events, fps):
        # 1. Vẽ các vùng cấm (ROI), highlight vùng đang bị xâm nhập trong frame này
        zones = config.get_roi_zones(camera_id)
        intruded_now = {
            evt["zone_name"]
            for evt in active_events if evt["event_type"] == "ZONE_INTRUSION"
        }
        for zone in zones:
            roi_pts = np.array(zone["polygon"], np.int32)
            color = (0, 0, 255) if zone["name"] not in intruded_now else (0, 165, 255)
            cv2.polylines(frame, [roi_pts], isClosed=True, color=color, thickness=3)
            cv2.putText(frame, zone["name"], (roi_pts[0][0], max(15, roi_pts[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. Vẽ Bounding Box & Track ID
        for t_id, obj in list(memory_manager.objects.items()):
            if len(obj.bbox_history) == 0:
                continue

            box = obj.bbox_history[-1]
            x1, y1, x2, y2 = map(int, box)

            color = (0, 255, 0) if obj.cls_id == 0 else (0, 255, 255)  # Green: Person, Yellow: Vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{t_id}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. Hiển thị Alerts
        y_offset = 60
        for evt in active_events:
            zone_txt = f" [{evt['zone_name']}]" if evt.get("zone_name") else ""
            alert_str = f"ALERT: {evt['event_type']}{zone_txt} (IDs: {evt['track_ids']})"
            cv2.putText(frame, alert_str, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25

        # 4. Hiển thị FPS & Cam ID
        cv2.putText(frame, f"Cam: {camera_id} | FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame