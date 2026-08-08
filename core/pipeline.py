import time
import cv2
import config
from core.frame_reader import ThreadedFrameReader
from core.visualizer import VideoVisualizer
from tracking.temporal_memory import TemporalMemoryManager
from events.classifier import RuleBasedEventClassifier


class CameraPipelineWorker:
    """Pipeline xử lý cho một camera: đọc frame -> detect/track -> temporal memory -> event -> MQTT."""

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
        self.visualizer = VideoVisualizer()
        self.frame_count = 0

        # Lưu frame mới nhất đã qua xử lý AI để hiển thị
        self.latest_display_frame = None
        self.fps = 0.0
        self.prev_time = time.time()

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

            # AI chạy theo SKIP_FRAMES
            if self.frame_count % config.SKIP_FRAMES == 0:
                active_events = self._process_ai(resized_frame)

            # Vẽ kết quả lên Frame
            self.latest_display_frame = self.visualizer.draw(
                resized_frame, self.camera_id, self.memory_manager, active_events, self.fps
            )

    def _process_ai(self, frame):
        """Detect + track + pose, cập nhật temporal memory và trả về events."""
        results = self.det_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )

        poses_dict = {}
        boxes = None

        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            cls_ids = boxes.cls.int().cpu().numpy()

            if 0 in cls_ids:
                pose_results = self.pose_model(frame, device=self.device, verbose=False)
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

        return active_events