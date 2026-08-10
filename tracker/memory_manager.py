from tracker.object_state import TrackedObjectState
import config


class ObjectMemoryManager:
    """Quản lý danh sách các object đang active và dọn dẹp các ID đã mất dấu."""

    def __init__(self, maxlen=30, ttl_frames=30):
        self.objects = {}  # track_id -> TrackedObjectState
        self.maxlen = maxlen
        self.ttl_frames = ttl_frames
        self.current_frame = 0

    def update_tracks(self, active_tracks, frame_idx, timestamp):
        self.current_frame = frame_idx
        updated_ids = set()

        for trk in active_tracks:
            t_id = trk["track_id"]
            bbox = trk["bbox"]
            cls_id = trk["cls_id"]
            pose = trk.get("pose", None)
            is_predicted = trk.get("predicted", False)

            if t_id not in self.objects:
                self.objects[t_id] = TrackedObjectState(t_id, cls_id, maxlen=self.maxlen)

            obj = self.objects[t_id]
            obj.last_update_predicted = is_predicted

            if is_predicted:
                # Track dự đoán: KHÔNG ghi vào history kinematics (velocity/accel/center/pose).
                # Chỉ lưu bbox để hiển thị. Velocity phải chỉ tính từ detection thật,
                # nếu không sẽ bị thổi phồng do dt giữa 2 frame detect cách nhau 3 frame.
                obj.predicted_bbox = bbox
                obj.missed_frames += 1
            else:
                obj.update(bbox, timestamp, pose, trk.get("vehicle_type"))
                obj.predicted_bbox = None
                obj.missed_frames = 0
                obj.last_updated_frame = frame_idx
            updated_ids.add(t_id)

        # Tăng missed_frames cho object không xuất hiện trong frame này
        for t_id, obj in self.objects.items():
            if t_id not in updated_ids:
                obj.missed_frames += 1

        # Cleanup lost objects
        dead_ids = [
            t_id for t_id, obj in self.objects.items()
            if self.current_frame - obj.last_updated_frame > self.ttl_frames
        ]
        for t_id in dead_ids:
            del self.objects[t_id]

    def visible_objects(self):
        """Chỉ trả về object còn bám dấu (detect thật gần đây); loại object ma.

        Cảnh báo chỉ tính trên object thật để không bị alert sớm/chậm hơn sự kiện.
        """
        return {
            t_id: obj for t_id, obj in self.objects.items()
            if len(obj.bbox_history) > 0 and obj.missed_frames < config.DETECTION_INTERVAL
        }