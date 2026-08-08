import numpy as np
import config


class PersonActionRules:
    """Xử lý ngã, vung tay mạnh, giằng co, xô xát dựa trên Temporal History & Pose."""

    def __init__(self):
        pass

    def check_fall(self, obj):
        """Phát hiện người ngã qua Aspect Ratio, Torso Angle & Gia tốc rơi."""
        if len(obj.bbox_history) < 3:
            return False

        # Check aspect ratio
        curr_bbox = obj.bbox_history[-1]
        w = curr_bbox[2] - curr_bbox[0]
        h = max(curr_bbox[3] - curr_bbox[1], 1e-5)
        aspect_ratio = w / h

        # Check Torso Angle từ Pose nếu có (Keypoint 5: L-Shoulder, 6: R-Shoulder, 11: L-Hip, 12: R-Hip)
        theta_torso = self._get_torso_angle(obj)

        # Check gia tốc rơi theo trục Y
        acc_y = [a[1] for a in list(obj.accel_history)[-5:]]
        max_v_y = max([v[1] for v in list(obj.velocity_history)[-5:]], default=0)

        is_horizontal = aspect_ratio > config.FALL_ASPECT_RATIO_THRESH or theta_torso > config.FALL_TORSO_ANGLE_THRESH
        is_falling_motion = max_v_y > 150.0 or any(a > 200.0 for a in acc_y)

        return is_horizontal and is_falling_motion

    def check_conflict(self, objA, objB):
        """Phát hiện 2-3 người tiếp cận, vung tay mạnh, giằng co, xô xát."""
        if len(objA.bbox_history) < 3 or len(objB.bbox_history) < 3:
            return False, 0.0

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        
        # BBox Height trung bình làm chuẩn khoảng cách
        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        avg_h = ((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2.0
        dist = np.linalg.norm(centA - centB) / max(avg_h, 1e-5)

        # Chỉ xét khi ở gần nhau (Proximity)
        if dist > config.CONFLICT_DIST_THRESH:
            return False, 0.0

        # Tốc độ vung tay / Động năng tay (Wrist Speed từ Pose)
        wrist_speed_A = self._get_wrist_speed(objA)
        wrist_speed_B = self._get_wrist_speed(objB)

        # Biến thiên khoảng cách liên tục (Interaction/Giằng co)
        dists = [
            np.linalg.norm(np.array(cA) - np.array(cB)) / max(avg_h, 1e-5)
            for cA, cB in zip(list(objA.center_history)[-5:], list(objB.center_history)[-5:])
        ]
        dist_variance = np.var(dists) if len(dists) > 1 else 0.0

        kinetic_score = wrist_speed_A + wrist_speed_B
        conflict_score = kinetic_score * 0.6 + dist_variance * 40.0 * 0.4

        return conflict_score > config.CONFLICT_KINETIC_THRESH, conflict_score

    def check_wild_gesture(self, obj):
        """Vung tay mạnh: wrist speed cao trong vài frame liên tiếp (Pose)."""
        if len(obj.pose_history) < 3:
            return False
        poses = list(obj.pose_history)
        times = list(obj.time_history)
        speeds = []
        for i in range(1, min(4, len(poses))):
            if poses[-i] is not None and poses[-i - 1] is not None:
                dt = max(times[-i] - times[-i - 1], 1e-5)
                speeds.append(self._wrist_speed_between(poses[-i - 1], poses[-i], dt))
        return max(speeds, default=0.0) > config.GESTURE_WRIST_SPEED_THRESH

    def check_person_collision(self, objA, objB):
        """2 người tiếp cận nhanh / va chạm dựa trên proximity + closing speed."""
        if len(objA.velocity_history) == 0 or len(objB.velocity_history) == 0:
            return False, 0.0

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        avg_h = ((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2.0
        dist = np.linalg.norm(centA - centB) / max(avg_h, 1e-5)

        if dist > config.PERSON_APPROACH_DIST_THRESH:
            return False, 0.0

        dir_vec = (centB - centA) / max(np.linalg.norm(centB - centA), 1e-5)
        velA = np.array(objA.velocity_history[-1])
        velB = np.array(objB.velocity_history[-1])
        closing_speed = float(np.dot(velA - velB, dir_vec))

        if closing_speed > config.PERSON_APPROACH_SPEED_THRESH:
            return True, min(1.0, closing_speed / 150.0)
        return False, 0.0

    def _get_torso_angle(self, obj):
        if len(obj.pose_history) == 0 or obj.pose_history[-1] is None:
            return 0.0
        kp = obj.pose_history[-1]  # Shape: (17, 3) [x, y, conf]
        if kp[5][2] > 0.3 and kp[6][2] > 0.3 and kp[11][2] > 0.3 and kp[12][2] > 0.3:
            neck = (kp[5][:2] + kp[6][:2]) / 2.0
            hip = (kp[11][:2] + kp[12][:2]) / 2.0
            vector = hip - neck
            angle = np.degrees(np.arctan2(abs(vector[0]), abs(vector[1])))
            return angle
        return 0.0

    def _get_wrist_speed(self, obj):
        if len(obj.pose_history) < 2:
            return 0.0
        kp_curr = obj.pose_history[-1]
        kp_prev = obj.pose_history[-2]
        if kp_curr is None or kp_prev is None:
            return 0.0
        dt = max(obj.time_history[-1] - obj.time_history[-2], 1e-5)
        return self._wrist_speed_between(kp_prev, kp_curr, dt)

    def _wrist_speed_between(self, kp_prev, kp_curr, dt):
        # Wrist Indices: 9 (Left), 10 (Right)
        speeds = []
        for idx in [9, 10]:
            if kp_curr[idx][2] > 0.3 and kp_prev[idx][2] > 0.3:
                spd = np.linalg.norm(kp_curr[idx][:2] - kp_prev[idx][:2]) / dt
                speeds.append(spd)
        return max(speeds, default=0.0)