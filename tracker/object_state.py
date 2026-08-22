from collections import deque
import numpy as np


class TrackedObjectState:
    """Lưu trữ lịch sử ngắn của đối tượng: BBox, Tâm, Vận tốc, Gia tốc, Pose.

    Bổ sung các bộ đếm persistence cho event detection:
    - fall_persist_count: đếm số frame liên tiếp ở tư thế ngã
    """

    def __init__(self, track_id, cls_id, maxlen=30):
        self.track_id = track_id
        self.cls_id = cls_id
        self.vehicle_type = None  # Loại xe tinh (xe máy, xe tải, xe chở dầu...)

        # Short Temporal Buffers
        self.bbox_history = deque(maxlen=maxlen)       # [x1, y1, x2, y2]
        self.center_history = deque(maxlen=maxlen)     # [cx, cy]
        self.velocity_history = deque(maxlen=maxlen)   # [vx, vy]
        self.accel_history = deque(maxlen=maxlen)      # [ax, ay]
        self.pose_history = deque(maxlen=maxlen)       # Keypoints (nếu có)
        self.time_history = deque(maxlen=maxlen)       # Timestamps

        self.dwell_times = {}  # {zone_name: frame_count}
        self.zone_inside = {}  # {zone_name: bool} trạng thái hiện diện trong vùng
        self.zone_entry = {}   # {zone_name: np.array} vị trí lúc vừa bước vào vùng
        self.last_updated_frame = 0
        self.missed_frames = 0  # Số frame liên tiếp không được detect thật (để dọn dấu cũ)
        self.last_update_predicted = False  # Frame gần nhất là dự đoán (không phải detection thật)
        self.predicted_bbox = None  # Bbox dự đoán hiển thị tạm; KHÔNG ghi vào history kinematics
        self.conf = 0.0  # Confidence detect gần nhất (gate collision pipeline Rule 6C)

        # Event Persistence Counters
        # Đếm số detection frame liên tiếp mà object ở trong trạng thái event.
        # Các rule sẽ tăng/giảm counter này, chỉ confirm khi vượt ngưỡng.
        self.fall_persist_count = 0

    def update(self, bbox, timestamp, pose=None, vehicle_type=None):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        center = np.array([cx, cy], dtype=np.float32)
        if vehicle_type is not None:
            self.vehicle_type = vehicle_type

        # 1. Tính Velocity & Acceleration
        if len(self.center_history) > 0:
            dt = max(timestamp - self.time_history[-1], 1e-5)
            vel = (center - self.center_history[-1]) / dt

            if len(self.velocity_history) > 0:
                acc = (vel - self.velocity_history[-1]) / dt
            else:
                acc = np.array([0.0, 0.0], dtype=np.float32)
        else:
            vel = np.array([0.0, 0.0], dtype=np.float32)
            acc = np.array([0.0, 0.0], dtype=np.float32)

        # Push to Deques
        self.bbox_history.append(bbox)
        self.center_history.append(center)
        self.velocity_history.append(vel)
        self.accel_history.append(acc)
        self.time_history.append(timestamp)
        if pose is not None:
            self.pose_history.append(pose)

    def update_zone_state(self, zone_name, is_inside, center=None):
        """Cập nhật trạng thái hiện diện trong vùng.

        Trả về True nếu đối tượng VỪA xâm nhập (trước đó ở ngoài) -> bước "Person enters ROI".
        Ghi lại vị trí lúc vào vùng (center) để bước "Movement" tính displacement không gian.
        """
        was_inside = self.zone_inside.get(zone_name, False)
        self.zone_inside[zone_name] = is_inside
        if is_inside and not was_inside:
            if center is not None:
                self.zone_entry[zone_name] = np.array(center, dtype=np.float32)
            return True
        if not is_inside:
            self.zone_entry.pop(zone_name, None)
        return False

    def tick_dwell(self, zone_name, is_inside):
        if is_inside:
            self.dwell_times[zone_name] = self.dwell_times.get(zone_name, 0) + 1
        else:
            self.dwell_times[zone_name] = 0