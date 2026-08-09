import os
import time
import argparse
import threading
import numpy as np
import cv2
import config
from tracker.memory_manager import ObjectMemoryManager
from events.classifier import RuleBasedEventClassifier
from events.visualizer import EventVisualizer
from detector import YOLODetector
from dashboard.store import AlertStore
from dashboard.server import start_dashboard


class CameraWorker(threading.Thread):
    """Worker Thread xử lý riêng từng Camera Stream (HLS/RTSP)."""

    def __init__(self, camera_id, stream_url, alert_store=None):
        super().__init__()
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.alert_store = alert_store
        self.memory_mgr = ObjectMemoryManager(maxlen=config.TEMPORAL_BUFFER_MAXLEN)
        self.event_classifier = RuleBasedEventClassifier()
        self.visualizer = EventVisualizer()
        self.detector = YOLODetector()
        self.stopped = False
        self.active_alert_keys = set()
        self.cls_names = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
        self._is_file_source = self._looks_like_file(stream_url)

    @staticmethod
    def _looks_like_file(source):
        """Phân biệt video local (file path) với stream HLS/RTSP (URL)."""
        if "://" in source:
            return False
        return os.path.exists(source)

    def _open_capture(self):
        if self._is_file_source:
            return cv2.VideoCapture(self.stream_url)
        return cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

    def run(self):
        headless = getattr(config, "HEADLESS", False)
        print(f"[{self.camera_id}] Source: {self.stream_url}"
              + (" (video local)" if self._is_file_source else " (HLS/RTSP stream)"))
        print(f"[{self.camera_id}] Mode: {'HEADLESS (log + snapshot)' if headless else 'GUI (cửa sổ video)'}")
        cap = self._open_capture()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        window_name = f"SmartVision AI Engine - {self.camera_id}"
        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0

        while not self.stopped and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                if self._is_file_source:
                    print(f"[{self.camera_id}] Hết video / không đọc được frame. Dừng xử lý.")
                    break
                print(f"[{self.camera_id}] Stream drop or buffering... Retrying in 1s")
                time.sleep(1.0)
                cap.open(self.stream_url, cv2.CAP_FFMPEG)
                continue

            frame_idx += 1
            timestamp = time.time()

            # Step 1: Preprocess Frame
            frame_resized = cv2.resize(frame, config.MODEL_INPUT_SIZE)

            # Step 2: Detection / Tracking
            # Detect mỗi DETECTION_INTERVAL frame, frame giữa dùng tracking dự đoán
            if frame_idx % config.DETECTION_INTERVAL == 0:
                active_tracks = self.detector.detect(frame_resized)
                if not headless:
                    counts = {}
                    for tr in active_tracks:
                        counts[tr["cls_id"]] = counts.get(tr["cls_id"], 0) + 1
                    desc = ", ".join(f"{self.cls_names.get(k, k)} x{v}" for k, v in counts.items()) or "none"
                    print(f"[{self.camera_id}] frame {frame_idx}: detected {desc}")
            else:
                active_tracks = self._predict_tracks(timestamp, frame_idx)

            # Step 3: Update History Memory
            self.memory_mgr.update_tracks(active_tracks, frame_idx, timestamp)

            # Step 4: Evaluate AI Vision Events
            events = self.event_classifier.evaluate(
                self.camera_id,
                self.memory_mgr,
                frame_resized,
                is_detection_frame=(frame_idx % config.DETECTION_INTERVAL == 0),
                frame_idx=frame_idx,
            )
            # Chỉ đưa alert có confidence đủ cao (giảm nhiễu / cảnh báo linh tinh)
            events = [ev for ev in events
                      if ev.get("confidence", 0.0) >= config.MIN_ALERT_CONFIDENCE]
            new_alert_events = self._log_events(events)

            # Step 5: Visual Overlay trực tiếp lên Video
            annotated_frame = self.visualizer.draw(
                frame_resized,
                self.memory_mgr,
                events,
                camera_id=self.camera_id,
            )

            # Step 6: Đưa ảnh alert lên dashboard trực tiếp (trong bộ nhớ)
            if self.alert_store is not None:
                for ev in new_alert_events:
                    alert = self.alert_store.record(self.camera_id, ev, annotated_frame)
                    if alert is not None:
                        print(f"[ALERT-IMG:{self.camera_id}] Đưa ảnh lên dashboard: "
                              f"{alert['snapshot']} (crop: {alert['crop']})")

            # Step 7: Render
            if headless:
                continue
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stopped = True
                break

        cap.release()
        cv2.destroyWindow(window_name)

    def _log_events(self, events):
        """Log alert khi event lần đầu được xác nhận (tránh spam lặp mỗi frame).

        Trả về danh sách event MỚI được xác nhận (để chụp snapshot dashboard).
        """
        new_events = []
        current_keys = set()
        for ev in events:
            key = (ev["event_type"], tuple(ev.get("track_ids", [])), ev.get("zone_name"))
            current_keys.add(key)
            if key not in self.active_alert_keys:
                self.active_alert_keys.add(key)
                conf = ev.get("confidence", 0.0)
                print(f"[ALERT:{self.camera_id}] {ev['event_type']} | {ev['description']} | conf={conf:.2f}")
                new_events.append(ev)

        # Giải phóng key không còn xuất hiện để event giống nhau có thể alert lại sau này
        for key in list(self.active_alert_keys):
            if key not in current_keys:
                self.active_alert_keys.discard(key)

        return new_events

    def _predict_tracks(self, timestamp, frame_idx):
        """Tracking ở frame giữa: dự đoán bbox theo vận tốc.

        Chỉ dự đoán object được detect thật gần đây (missed_frames nhỏ).
        Object đã mất dấu sẽ ngừng vẽ -> đánh dấu cũ được xoá khi sang frame mới.
        """
        tracks = []
        for t_id, obj in self.memory_mgr.objects.items():
            if len(obj.bbox_history) == 0 or len(obj.velocity_history) == 0:
                continue
            if obj.missed_frames >= config.DETECTION_INTERVAL:
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
                "predicted": True,
            })
        return tracks


def main():
    print("============================================================")
    print(" SMARTVISION AI ENGINE - SINGLE SOURCE PER RUN")
    print("============================================================")

    parser = argparse.ArgumentParser(
        description="SmartVision AI - xử lý 1 camera/video tại một thời điểm."
    )
    parser.add_argument(
        "camera_id",
        nargs="?",
        help="ID camera cần chạy (ví dụ: cam09). Xem danh sách trong config.CAMERA_STREAMS.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Đường dẫn video local để phân tích (nếu không có, dùng stream HLS của camera).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Chạy không cửa sổ video: chỉ log cảnh báo + chụp snapshot vị trí cảnh báo.",
    )
    args = parser.parse_args()

    if not args.camera_id:
        print("[SYSTEM] Chưa chọn camera. Chạy riêng từng camera (mỗi lần 1 nguồn):")
        for cid in config.CAMERA_STREAMS:
            print(f"    python main.py {cid}")
        print("[SYSTEM] Hoặc phân tích 1 video local bằng camera config nào đó:")
        print("    python main.py cam09 --video path/to/video.mp4")
        return

    if args.camera_id not in config.CAMERA_STREAMS:
        print(f"[SYSTEM] Không tìm thấy camera '{args.camera_id}'. "
              f"Các camera có sẵn: {list(config.CAMERA_STREAMS.keys())}")
        return

    if args.video and not os.path.exists(args.video):
        print(f"[SYSTEM] Không tìm thấy file video: {args.video}")
        return

    source = args.video or config.CAMERA_STREAMS[args.camera_id]

    # Bật headless mode qua CLI (ghi đè config)
    if args.headless:
        config.HEADLESS = True

    # 1. Khởi chạy AlertStore + Dashboard Web (nếu bật)
    alert_store = AlertStore(
        snapshot_dir=config.SNAPSHOT_DIR,
        max_alerts=config.SNAPSHOT_MAX_ALERTS,
    )
    start_dashboard(alert_store, config.SNAPSHOT_DIR, enabled=config.DASHBOARD_ENABLED)

    # 2. Khởi chạy DUY NHẤT worker của camera được chọn
    worker = CameraWorker(args.camera_id, source, alert_store=alert_store)
    worker.daemon = True
    worker.start()

    try:
        while worker.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[SYSTEM] Stopping pipeline...")
        worker.stopped = True


if __name__ == "__main__":
    main()
