import time
import threading
import numpy as np
import cv2
import config
from tracker.memory_manager import ObjectMemoryManager
from events.classifier import RuleBasedEventClassifier
from events.visualizer import EventVisualizer
from sensors.mqtt_consumer import SensorMQTTConsumer
from detector import YOLODetector


class CameraWorker(threading.Thread):
    """Worker Thread xử lý riêng từng Camera Stream (HLS/RTSP)."""

    def __init__(self, camera_id, stream_url, sensor_consumer):
        super().__init__()
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.sensor_consumer = sensor_consumer
        self.memory_mgr = ObjectMemoryManager(maxlen=config.TEMPORAL_BUFFER_MAXLEN)
        self.event_classifier = RuleBasedEventClassifier()
        self.visualizer = EventVisualizer()
        self.detector = YOLODetector()
        self.stopped = False
        self.active_alert_keys = set()
        self.cls_names = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

    def run(self):
        print(f"[{self.camera_id}] Connecting to stream: {self.stream_url}")
        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        window_name = f"SmartVision AI Engine - {self.camera_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0

        while not self.stopped and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[{self.camera_id}] Stream drop or buffering... Retrying in 1s")
                time.sleep(1.0)
                cap.open(self.stream_url, cv2.CAP_FFMPEG)
                continue

            frame_idx += 1
            timestamp = time.time()

            # Step 1: Preprocess Frame
            frame_resized = cv2.resize(frame, config.MODEL_INPUT_SIZE)

            # Step 2: Detection / Tracking
            # Detect mỗi DETECTION_INTERVAL frame, frame giữa dùng tracking dự đoán (Kalman/Constant Velocity)
            if frame_idx % config.DETECTION_INTERVAL == 0:
                active_tracks = self.detector.detect(frame_resized)
                counts = {}
                for tr in active_tracks:
                    counts[tr["cls_id"]] = counts.get(tr["cls_id"], 0) + 1
                desc = ", ".join(f"{self.cls_names.get(k, k)} x{v}" for k, v in counts.items()) or "none"
                print(f"[{self.camera_id}] frame {frame_idx}: detected {desc}")
            else:
                active_tracks = self._predict_tracks(timestamp)

            # Step 3: Update History Memory
            self.memory_mgr.update_tracks(active_tracks, frame_idx, timestamp)

            # Step 4: Evaluate AI Vision Events
            events = self.event_classifier.evaluate(self.camera_id, self.memory_mgr, frame_resized)
            self._log_events(events)

            # Step 5: Lấy Sensor Alerts từ MQTT Consumer
            sensor_alerts = self.sensor_consumer.get_latest_alerts()

            # Step 6: Visual Overlay trực tiếp lên Video
            annotated_frame = self.visualizer.draw(
                frame_resized,
                self.memory_mgr,
                events,
                sensor_alerts=sensor_alerts,
                camera_id=self.camera_id,
            )

            # Step 7: Render
            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stopped = True
                break

        cap.release()
        cv2.destroyWindow(window_name)

    def _log_events(self, events):
        """Log alert khi event lần đầu được xác nhận (tránh spam lặp mỗi frame)."""
        current_keys = set()
        for ev in events:
            key = (ev["event_type"], tuple(ev.get("track_ids", [])), ev.get("zone_name"))
            current_keys.add(key)
            if key not in self.active_alert_keys:
                self.active_alert_keys.add(key)
                conf = ev.get("confidence", 0.0)
                print(f"[ALERT:{self.camera_id}] {ev['event_type']} | {ev['description']} | conf={conf:.2f}")

        # Giải phóng key không còn xuất hiện để event giống nhau có thể alert lại sau này
        for key in list(self.active_alert_keys):
            if key not in current_keys:
                self.active_alert_keys.discard(key)

    def _predict_tracks(self, timestamp):
        """Tracking ở frame giữa: dùng vận tốc gần nhất để dự đoán bbox tiếp theo."""
        tracks = []
        for t_id, obj in self.memory_mgr.objects.items():
            if len(obj.bbox_history) == 0 or len(obj.velocity_history) == 0:
                continue
            dt = max(timestamp - obj.time_history[-1], 1e-5)
            vel = obj.velocity_history[-1]
            last_box = np.array(obj.bbox_history[-1], dtype=np.float32)
            pred = last_box + np.array([vel[0], vel[1], vel[0], vel[1]]) * dt
            pred = np.clip(pred, 0, 2000)
            if not np.all(np.isfinite(pred)):
                continue
            tracks.append({
                "track_id": t_id,
                "cls_id": obj.cls_id,
                "bbox": pred.tolist(),
                "conf": 0.0,
                "pose": obj.pose_history[-1] if obj.pose_history else None,
            })
        return tracks


def main():
    print("============================================================")
    print(" SMARTVISION AI ENGINE - MULTI-STREAM & SENSOR INTEGRATION")
    print("============================================================")

    # 1. Khởi chạy Sensor MQTT Consumer
    sensor_consumer = SensorMQTTConsumer()
    sensor_consumer.start()

    # 2. Khởi chạy đồng thời 2 luồng Camera: Cam09 & Cam10
    workers = []
    for cam_id, url in config.CAMERA_STREAMS.items():
        worker = CameraWorker(cam_id, url, sensor_consumer)
        worker.daemon = True
        worker.start()
        workers.append(worker)

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[SYSTEM] Stopping pipeline...")
        for w in workers:
            w.stopped = True


if __name__ == "__main__":
    main()