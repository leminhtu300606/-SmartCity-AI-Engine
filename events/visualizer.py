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
        for t_id, obj in memory_manager.visible_objects().items():
            latest_box = obj.predicted_bbox if obj.last_update_predicted else obj.bbox_history[-1]
            box = self._to_int_bbox(latest_box)
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

        # 2b. Vẽ lưới ô nhỏ (grid) cho phát hiện khói/lửa theo từng điểm
        if config.SMOKE_FIRE_GRID_ENABLED:
            h, w = canvas.shape[:2]
            step_x = w / config.SMOKE_FIRE_GRID_COLS
            step_y = h / config.SMOKE_FIRE_GRID_ROWS
            grid_color = (100, 100, 100)
            for i in range(1, config.SMOKE_FIRE_GRID_COLS):
                x = int(round(i * step_x))
                cv2.line(canvas, (x, 0), (x, h), grid_color, 1)
            for j in range(1, config.SMOKE_FIRE_GRID_ROWS):
                y = int(round(j * step_y))
                cv2.line(canvas, (0, y), (w, y), grid_color, 1)

        # 3. Gom chung AI Events & Sensor Alerts (dedup: chỉ giữ 1 cảnh báo mỗi loại)
        all_alerts = []
        seen_types = set()

        # 3b. Vẽ bbox VỊ TRÍ nghi vấn lên khung hình (trước khi dedup banner)
        #     Lửa/khói theo ô: bbox = ô grid. Đánh nhau/va chạm: bbox = vùng 2 đối tượng.
        for ev in list(active_events):
            box = self._to_int_bbox(ev.get("bbox"))
            if box is not None:
                if ev["event_type"] in ("FIRE_DETECTED", "SMOKE_DETECTED"):
                    color = (0, 0, 255)  # đỏ cho lửa/khói
                elif ev["event_type"] == "HUMAN_CONFLICT":
                    color = (0, 0, 255)
                else:
                    color = (0, 165, 255)  # cam cho va chạm/ngã/xâm nhập
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(
                    canvas,
                    ev["event_type"],
                    (box[0], box[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        for ev in list(active_events):
            if ev.get("event_type") not in seen_types:
                seen_types.add(ev.get("event_type"))
                all_alerts.append(ev)
        if sensor_alerts:
            for sa in sensor_alerts:
                if sa["event_type"] not in seen_types:
                    seen_types.add(sa["event_type"])
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