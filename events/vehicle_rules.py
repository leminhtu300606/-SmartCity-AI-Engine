import numpy as np


class VehicleAccidentRules:
    """Xử lý va chạm xe, vật thể rơi vào, dừng bất thường dựa trên Trajectory, Velocity, Deceleration."""

    def check_collision(self, objA, objB):
        """Phát hiện va chạm bằng BBox overlap/proximity + Giảm tốc/Đổi hướng đột ngột."""
        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        
        # 1. BBox Overlap (IoU)
        iou = self._calculate_iou(boxA, boxB)
        
        # 2. Distance Proximity
        centA, centB = np.array(objA.center_history[-1]), np.array(objB.center_history[-1])
        dist = np.linalg.norm(centA - centB)

        # 3. Sudden Deceleration & Direction Change (Không dùng đơn thuần bbox overlap)
        decelA = self._get_deceleration(objA)
        decelB = self._get_deceleration(objB)
        dir_changeA = self._get_direction_change(objA)
        dir_changeB = self._get_direction_change(objB)

        # Xe tai nạn / dừng bất thường / đổi hướng ngột ngạt sau tiếp cận
        is_proximate = iou > 0.05 or dist < 50.0
        is_kinematic_anomaly = (decelA > 100.0 or decelB > 100.0) or (dir_changeA > 1.0 or dir_changeB > 1.0)

        return is_proximate and is_kinematic_anomaly

    def check_hard_stop(self, obj):
        """Phát hiện xe tai nạn / dừng bất thường giữa đường."""
        if len(obj.velocity_history) < 5:
            return False
        speeds = [np.linalg.norm(v) for v in list(obj.velocity_history)[-5:]]
        is_stopped = speeds[-1] < 2.0
        had_high_speed = speeds[0] > 30.0
        return is_stopped and had_high_speed

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def _get_deceleration(self, obj):
        if len(obj.accel_history) < 2:
            return 0.0
        acc = obj.accel_history[-1]
        vel = obj.velocity_history[-1]
        # Bỏ qua nếu xe đang đứng yên
        if np.linalg.norm(vel) < 1e-3:
            return 0.0
        # Chiếu gia tốc ngược hướng vận tốc = Giảm tốc
        decel = -np.dot(acc, vel) / np.linalg.norm(vel)
        return max(0.0, decel)

    def _get_direction_change(self, obj):
        if len(obj.direction_history) < 3:
            return 0.0
        dirs = list(obj.direction_history)
        delta = dirs[-1] - dirs[-3]
        # Normalize angular difference to [0, pi] (handle +/- pi seam)
        return abs(np.arctan2(np.sin(delta), np.cos(delta)))