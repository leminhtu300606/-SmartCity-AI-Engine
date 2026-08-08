from collections import deque
import numpy as np

class TrackedObjectState:
    """Lưu lịch sử ngắn (1.0 - 1.5s) của từng đối tượng để phân tích temporal motion"""
    def __init__(self, track_id, cls_id, max_history=30):
        self.track_id = track_id
        self.cls_id = cls_id
        self.bbox_history = deque(maxlen=max_history)    # [x1, y1, x2, y2]
        self.center_history = deque(maxlen=max_history)  # (cx, cy)
        self.pose_history = deque(maxlen=max_history)    # Keypoints (17, 3)
        self.velocity_history = deque(maxlen=max_history)# (vx, vy)
        self.dwell_time = 0                              # Số frame ở trong vùng cấm

    def update(self, bbox, keypoints=None, dt=0.04):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        new_center = (cx, cy)

        if len(self.center_history) > 0:
            prev_center = self.center_history[-1]
            vx = (new_center[0] - prev_center[0]) / dt
            vy = (new_center[1] - prev_center[1]) / dt
            self.velocity_history.append((vx, vy))

        self.bbox_history.append(bbox)
        self.center_history.append(new_center)
        if keypoints is not None:
            self.pose_history.append(keypoints)

class TemporalMemoryManager:
    def __init__(self, max_history=30):
        self.objects = {} # track_id -> TrackedObjectState
        self.max_history = max_history

    def update_tracks(self, boxes, poses_dict=None, dt=0.04):
        """
        boxes: Ultralytics Results[0].boxes
        """
        if boxes is None or boxes.id is None:
            return

        # Chuyển đổi tensor sang numpy array
        track_ids = boxes.id.int().cpu().numpy()
        cls_ids = boxes.cls.int().cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()

        active_ids = set()
        for idx, t_id in enumerate(track_ids):
            t_id = int(t_id)
            cls_id = int(cls_ids[idx])
            bbox = xyxys[idx] # [x1, y1, x2, y2]

            kpts = poses_dict.get(t_id, None) if poses_dict else None

            if t_id not in self.objects:
                self.objects[t_id] = TrackedObjectState(t_id, cls_id, self.max_history)
            
            self.objects[t_id].update(bbox, kpts, dt)
            active_ids.add(t_id)

        # Xóa các track_id lâu không xuất hiện
        missing_ids = set(self.objects.keys()) - active_ids
        for m_id in missing_ids:
            del self.objects[m_id]