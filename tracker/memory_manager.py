from tracker.object_state import TrackedObjectState


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

            if t_id not in self.objects:
                self.objects[t_id] = TrackedObjectState(t_id, cls_id, maxlen=self.maxlen)

            obj = self.objects[t_id]
            obj.update(bbox, timestamp, pose)
            obj.last_updated_frame = frame_idx
            updated_ids.add(t_id)

        # Cleanup lost objects
        dead_ids = [
            t_id for t_id, obj in self.objects.items()
            if self.current_frame - obj.last_updated_frame > self.ttl_frames
        ]
        for t_id in dead_ids:
            del self.objects[t_id]