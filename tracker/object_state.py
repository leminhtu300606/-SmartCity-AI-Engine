from collections import deque
import numpy as np


class TrackedObjectState:
    """Lưu trữ lịch sử ngắn của đối tượng: BBox, Tâm, Vận tốc, Gia tốc, Pose."""

    def __init__(self, track_id, cls_id, maxlen=30):
        self.track_id = track_id
        self.cls_id = cls_id
        
        # Short Temporal Buffers
        self.bbox_history = deque(maxlen=maxlen)       # [x1, y1, x2, y2]
        self.center_history = deque(maxlen=maxlen)     # [cx, cy]
        self.velocity_history = deque(maxlen=maxlen)   # [vx, vy]
        self.accel_history = deque(maxlen=maxlen)      # [ax, ay]
        self.direction_history = deque(maxlen=maxlen)  # Radian / Angle
        self.pose_history = deque(maxlen=maxlen)       # Keypoints (nếu có)
        self.time_history = deque(maxlen=maxlen)       # Timestamps

        self.dwell_times = {}  # {zone_name: frame_count}
        self.last_updated_frame = 0

    def update(self, bbox, timestamp, pose=None):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        center = np.array([cx, cy], dtype=np.float32)

        # 1. Tính Velocity & Acceleration
        if len(self.center_history) > 0:
            dt = max(timestamp - self.time_history[-1], 1e-5)
            vel = (center - self.center_history[-1]) / dt
            
            if len(self.velocity_history) > 0:
                acc = (vel - self.velocity_history[-1]) / dt
            else:
                acc = np.array([0.0, 0.0], dtype=np.float32)

            direction = np.arctan2(vel[1], vel[0])
        else:
            vel = np.array([0.0, 0.0], dtype=np.float32)
            acc = np.array([0.0, 0.0], dtype=np.float32)
            direction = 0.0

        # Push to Deques
        self.bbox_history.append(bbox)
        self.center_history.append(center)
        self.velocity_history.append(vel)
        self.accel_history.append(acc)
        self.direction_history.append(direction)
        self.time_history.append(timestamp)
        if pose is not None:
            self.pose_history.append(pose)

    def tick_dwell(self, zone_name, is_inside):
        if is_inside:
            self.dwell_times[zone_name] = self.dwell_times.get(zone_name, 0) + 1
        else:
            self.dwell_times[zone_name] = 0