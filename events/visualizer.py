import cv2
import numpy as np
import config


class EventVisualizer:
    """Vẽ Visual ROI, Bounding Box, Pose & Banner Alert trực tiếp lên hình ảnh Video."""

    def draw(self, frame, memory_manager, active_events, sensor_alerts=None, camera_id="cam09"):
        canvas = frame.copy()

        # 1. Vẽ các Polygon ROI
        zones = config.CAMERA_ROIS.get(camera_id, [])
        for zone in zones:
            pts = np.array(zone["polygon"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(
                canvas,
                zone["name"],
                (zone["polygon"][0][0], zone["polygon"][0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        # 2. Vẽ Track Bounding Boxes (green: person, yellow: vehicle) + HUD đếm
        cls_names = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
        class_counts = {}
        for t_id, obj in memory_manager.objects.items():
            if len(obj.bbox_history) == 0:
                continue
            box = self._to_int_bbox(obj.bbox_history[-1])
            if box is None:
                continue
            class_counts[obj.cls_id] = class_counts.get(obj.cls_id, 0) + 1
            color = (0, 255, 0) if obj.cls_id == 0 else (0, 255, 255)
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(
                canvas,
                f"ID:{t_id} {cls_names.get(obj.cls_id, obj.cls_id)}",
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # HUD: số object theo class
        hud = " | ".join(f"{cls_names.get(k, k)}:{v}" for k, v in class_counts.items()) or "no object"
        cv2.putText(canvas, f"{camera_id} | {hud}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # 3. Gom chung AI Events & Sensor Alerts
        all_alerts = list(active_events)
        if sensor_alerts:
            for sa in sensor_alerts:
                all_alerts.append({
                    "event_type": sa["event_type"],
                    "description": f"[{sa['device_id']}] {sa['description']}",
                    "confidence": 1.0
                })

        # 4. Vẽ Banner Cảnh báo phủ lên phía trên Video
        if all_alerts:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 40 * len(all_alerts)), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

            for idx, ev in enumerate(all_alerts):
                alert_text = f"[ALERT] {ev['event_type']} - {ev['description']}"
                cv2.putText(
                    canvas,
                    alert_text,
                    (10, 25 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

        return canvas

    @staticmethod
    def _to_int_bbox(box):
        """Chuẩn hóa bbox về list 4 số int; bỏ qua nếu NaN/inf hoặc sai độ dài."""
        vals = []
        try:
            for v in np.ravel(box):
                f = float(v)
                if not np.isfinite(f):
                    return None
                vals.append(int(round(f)))
        except (TypeError, ValueError):
            return None
        if len(vals) != 4:
            return None
        return vals