import numpy as np
import config


class VehicleAccidentRules:
    """Phát hiện va chạm xe, dừng bất thường dựa trên:
    - Khoảng cách chuẩn hóa (theo kích thước xe, không dùng pixel tuyệt đối)
    - Tốc độ tiến lại gần nhau (closing velocity)
    - Giảm tốc / đổi hướng đột ngột
    - Temporal sustained confirmation (không dùng đơn thuần bbox overlap = collision)
    """

    def __init__(self):
        # Temporal state cho sustained confirmation
        self.collision_state = {}    # (min_id, max_id) -> sustained_count
        self.hard_stop_state = {}    # track_id -> sustained_count

    def check_collision(self, objA, objB):
        """Phát hiện va chạm bằng multi-signal analysis + temporal confirmation."""
        if len(objA.bbox_history) < 3 or len(objB.bbox_history) < 3:
            return False

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]

        # 1. Normalized proximity (theo đường chéo trung bình của xe)
        diagA = np.sqrt((boxA[2] - boxA[0]) ** 2 + (boxA[3] - boxA[1]) ** 2)
        diagB = np.sqrt((boxB[2] - boxB[0]) ** 2 + (boxB[3] - boxB[1]) ** 2)
        avg_diag = max((diagA + diagB) / 2.0, 1e-5)

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        dist = np.linalg.norm(centA - centB)
        rel_dist = dist / avg_diag

        # 2. IoU
        iou = self._calculate_iou(boxA, boxB)

        # 3. Closing velocity
        closing_speed = self._get_closing_speed(objA, objB)

        # 4. Kinematic anomaly (giảm tốc / đổi hướng đột ngột)
        decelA = self._get_deceleration(objA)
        decelB = self._get_deceleration(objB)
        dir_changeA = self._get_direction_change(objA)
        dir_changeB = self._get_direction_change(objB)

        # Multi-signal scoring
        is_proximate = (iou > config.VEHICLE_IOU_THRESH
                        or rel_dist < config.VEHICLE_PROXIMITY_DIST_RATIO)
        is_kinematic = (decelA > config.VEHICLE_DECEL_THRESH
                        or decelB > config.VEHICLE_DECEL_THRESH
                        or dir_changeA > config.VEHICLE_DIR_CHANGE_THRESH
                        or dir_changeB > config.VEHICLE_DIR_CHANGE_THRESH)
        is_closing = closing_speed > config.VEHICLE_CLOSING_SPEED_THRESH

        # Cần proximity + (kinematic anomaly HOẶC closing speed cao)
        is_candidate = is_proximate and (is_kinematic or is_closing)

        # Temporal sustained confirmation
        pair_key = (min(objA.track_id, objB.track_id),
                    max(objA.track_id, objB.track_id))
        if is_candidate:
            self.collision_state[pair_key] = self.collision_state.get(pair_key, 0) + 1
        else:
            self.collision_state[pair_key] = max(
                0, self.collision_state.get(pair_key, 0) - 1
            )

        return self.collision_state.get(pair_key, 0) >= config.VEHICLE_COLLISION_SUSTAINED

    def check_hard_stop(self, obj):
        """Phát hiện xe dừng bất thường với temporal confirmation.

        Kiểm tra: đang chạy nhanh → dừng đột ngột (có giai đoạn giảm tốc mạnh).
        """
        history_len = config.VEHICLE_HARD_STOP_HISTORY
        if len(obj.velocity_history) < history_len:
            return False

        speeds = [np.linalg.norm(v) for v in list(obj.velocity_history)[-history_len:]]

        # Tốc độ trong nửa đầu cửa sổ phải cao, nửa cuối phải thấp
        half = history_len // 2
        max_prior_speed = max(speeds[:half])
        current_speed = min(speeds[-3:])  # Vài frame cuối phải đều chậm

        is_stopped = (current_speed < config.VEHICLE_HARD_STOP_SPEED_LOW
                      and max_prior_speed > config.VEHICLE_HARD_STOP_SPEED_HIGH)

        # Kiểm tra có giai đoạn giảm tốc mạnh trong cửa sổ
        has_sharp_decel = False
        for i in range(1, len(speeds)):
            if speeds[i - 1] > 15.0 and speeds[i] < 5.0:
                has_sharp_decel = True
                break

        is_candidate = is_stopped and has_sharp_decel

        # Sustained confirmation
        tid = obj.track_id
        if is_candidate:
            self.hard_stop_state[tid] = self.hard_stop_state.get(tid, 0) + 1
        else:
            self.hard_stop_state[tid] = max(0, self.hard_stop_state.get(tid, 0) - 1)

        return self.hard_stop_state.get(tid, 0) >= config.VEHICLE_HARD_STOP_SUSTAINED

    # ----------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------

    def _get_closing_speed(self, objA, objB):
        """Tốc độ tiến lại gần nhau (chiếu relative velocity lên đường nối 2 tâm)."""
        if len(objA.velocity_history) == 0 or len(objB.velocity_history) == 0:
            return 0.0

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        direction = centB - centA
        dist = np.linalg.norm(direction)

        velA = np.array(objA.velocity_history[-1])
        velB = np.array(objB.velocity_history[-1])
        rel_vel = velA - velB

        if dist < 1e-5:
            # Hai tâm trùng nhau (thời điểm va chạm): dùng hướng vận tốc tương đối
            # làm fallback, không để closing speed tụt về 0 giữa lúc tiếp xúc.
            rv_norm = np.linalg.norm(rel_vel)
            if rv_norm < 1e-5:
                return 0.0
            direction = rel_vel / rv_norm
        else:
            direction = direction / dist

        # Closing speed = relative velocity projected onto connecting line
        # Dương = đang tiến lại gần nhau
        closing = float(np.dot(rel_vel, direction))
        return max(0.0, closing)

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

    def cleanup_lost_tracks(self, active_track_ids):
        """Xoá state cho các track đã mất dấu."""
        lost_pairs = [k for k in self.collision_state
                      if k[0] not in active_track_ids or k[1] not in active_track_ids]
        for k in lost_pairs:
            del self.collision_state[k]

        lost_singles = [k for k in self.hard_stop_state if k not in active_track_ids]
        for k in lost_singles:
            del self.hard_stop_state[k]