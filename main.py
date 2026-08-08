import time
import threading
import cv2
import torch
import numpy as np
from ultralytics import YOLO

import config
from stream_reader import ThreadedFrameReader
from temporal_memory import TemporalMemoryManager
from event_classifier import RuleBasedEventClassifier
from mqtt_publisher import MQTTPublisher

class CameraPipelineWorker:
    def __init__(self, camera_id, stream_url, det_model, pose_model, mqtt_pub, device):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.det_model = det_model
        self.pose_model = pose_model
        self.mqtt_pub = mqtt_pub
        self.device = device

        self.reader = ThreadedFrameReader(camera_id, stream_url)
        self.memory_manager = TemporalMemoryManager(max_history=30)
        self.event_classifier = RuleBasedEventClassifier(fps=25)
        self.frame_count = 0
        
        # Lưu frame mới nhất đã qua xử lý AI để hiển thị
        self.latest_display_frame = None
        self.fps = 0.0
        self.prev_time = time.time()

    def draw_visualizations(self, frame, active_events):
        # 1. Vẽ Vùng cấm (ROI) nếu cấu hình có ROI cho camera này
        if self.camera_id in config.RESTRICTED_ROIS and len(config.RESTRICTED_ROIS[self.camera_id]) > 0:
            roi_pts = np.array(config.RESTRICTED_ROIS[self.camera_id], np.int32)
            cv2.polylines(frame, [roi_pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, "RESTRICTED ZONE", (roi_pts[0][0], max(15, roi_pts[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 2. Vẽ Bounding Box & Track ID
        for t_id, obj in list(self.memory_manager.objects.items()):
            if len(obj.bbox_history) == 0:
                continue
            
            box = obj.bbox_history[-1]
            x1, y1, x2, y2 = map(int, box)
            
            color = (0, 255, 0) if obj.cls_id == 0 else (0, 255, 255) # Green: Person, Yellow: Vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{t_id}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. Hiển thị Alerts
        y_offset = 60
        for evt in active_events:
            alert_str = f"ALERT: {evt['event_type']} (IDs: {evt['track_ids']})"
            cv2.putText(frame, alert_str, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25

        # 4. Hiển thị FPS & Cam ID
        cv2.putText(frame, f"Cam: {self.camera_id} | FPS: {self.fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame

    def run(self):
        self.reader.start()

        while True:
            ret, frame = self.reader.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            self.frame_count += 1
            curr_time = time.time()
            self.fps = 1.0 / max(curr_time - self.prev_time, 1e-5)
            self.prev_time = curr_time

            resized_frame = cv2.resize(frame, config.INFERENCE_SIZE)
            active_events = []

            # AI Chạy theo SKIP_FRAMES
            if self.frame_count % config.SKIP_FRAMES == 0:
                results = self.det_model.track(
                    resized_frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=self.device,
                    verbose=False
                )

                poses_dict = {}
                boxes = None

                if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    cls_ids = boxes.cls.int().cpu().numpy()

                    if 0 in cls_ids:
                        pose_results = self.pose_model(resized_frame, device=self.device, verbose=False)
                        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                            kpts_data = pose_results[0].keypoints.data.cpu().numpy()
                            person_idx = 0
                            for idx, c_id in enumerate(cls_ids):
                                if c_id == 0 and person_idx < len(kpts_data):
                                    track_id = int(boxes.id[idx].item())
                                    poses_dict[track_id] = kpts_data[person_idx]
                                    person_idx += 1

                if boxes is not None:
                    self.memory_manager.update_tracks(boxes, poses_dict)

                active_events = self.event_classifier.evaluate_events(self.camera_id, self.memory_manager)

                for evt in active_events:
                    self.mqtt_pub.publish_event(self.camera_id, evt)

            # Vẽ kết quả lên Frame
            self.latest_display_frame = self.draw_visualizations(resized_frame, active_events)


def main():
    print("=" * 60)
    print("  SMARTVISION AI ENGINE PHASE 1 - SMOOTH GUI DISPLAY ")
    print("=" * 60)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Hardware Acceleration Device: {device.upper()}")

    det_model = YOLO(config.MODEL_DET_PATH)
    pose_model = YOLO(config.MODEL_POSE_PATH)
    mqtt_pub = MQTTPublisher()

    workers = {}
    for cam_id, stream_url in config.CAMERA_STREAMS.items():
        worker = CameraPipelineWorker(cam_id, stream_url, det_model, pose_model, mqtt_pub, device)
        t = threading.Thread(target=worker.run, daemon=True)
        t.start()
        workers[cam_id] = worker

    print(f"[SYSTEM] Running pipelines for {len(workers)} cameras.")
    print("[SYSTEM] Press 'q' on any video window to exit.")

    # Luồng chính (Main Thread) chuyên làm nhiệm vụ render hiển thị video
    try:
        while True:
            for cam_id, worker in workers.items():
                if worker.latest_display_frame is not None:
                    cv2.imshow(f"SmartVision AI - {cam_id}", worker.latest_display_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\n[SYSTEM] Stopped successfully.")

if __name__ == "__main__":
    main()