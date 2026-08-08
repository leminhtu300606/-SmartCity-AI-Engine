import numpy as np
import config

class PersonActionRules:
    """Rules cho hành vi bất thường của người: ngã (fall) và xô xát (conflict)."""
    def __init__(self, dt):
        self.dt = dt

    def check_fall(self, obj):
        if len(obj.bbox_history) < 10:
            return False

        curr_bbox = obj.bbox_history[-1]
        w = curr_bbox[2] - curr_bbox[0]
        h = max(curr_bbox[3] - curr_bbox[1], 1e-5)
        aspect_ratio = w / h

        centers_y = [c[1] for c in obj.center_history]
        heights = [max(b[3] - b[1], 1e-5) for b in obj.bbox_history]
        v_y = np.diff(centers_y) / (np.array(heights[:-1]) * self.dt)
        max_v_y = np.max(v_y) if len(v_y) > 0 else 0.0

        theta_torso = self._get_torso_angle(obj)

        cond_drop = max_v_y > config.FALL_V_Y_THRESH
        cond_lying = (aspect_ratio > config.FALL_ASPECT_RATIO_THRESH) or (theta_torso > config.FALL_TORSO_ANGLE_THRESH)
        return cond_drop and cond_lying

    def check_conflict(self, objA, objB):
        """Kiểm tra xô xát giữa cặp người. Trả về (is_fight, score)."""
        if len(objA.bbox_history) < 10 or len(objB.bbox_history) < 10:
            return False, 0.0

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        centA, centB = np.array(objA.center_history[-1]), np.array(objB.center_history[-1])
        avg_h = ((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2.0
        dist = np.linalg.norm(centA - centB) / max(avg_h, 1e-5)

        if dist > config.CONFLICT_GATING_DIST:
            return False, 0.0

        speedA = self._get_wrist_speed(objA)
        speedB = self._get_wrist_speed(objB)
        kinetic_score = min((speedA + speedB) / 8.0, 1.0)

        dists = [np.linalg.norm(np.array(cA) - np.array(cB)) / max(avg_h, 1e-5)
                 for cA, cB in zip(objA.center_history, objB.center_history)]
        var_score = min(np.var(dists) * 10.0, 1.0)

        score = 0.7 * kinetic_score + 0.3 * var_score
        return score > config.CONFLICT_SCORE_THRESH, score

    def _get_torso_angle(self, obj):
        """Góc nghiêng thân người (độ) từ keypoint vai (5,6) và hông (11,12)."""
        if len(obj.pose_history) == 0:
            return 0.0
        curr_pose = obj.pose_history[-1]
        if curr_pose[5][2] > 0.3 and curr_pose[6][2] > 0.3 and curr_pose[11][2] > 0.3 and curr_pose[12][2] > 0.3:
            neck = (curr_pose[5][:2] + curr_pose[6][:2]) / 2.0
            hip = (curr_pose[11][:2] + curr_pose[12][:2]) / 2.0
            u = hip - neck
            norm_u = np.linalg.norm(u)
            if norm_u > 0:
                cos_theta = u[1] / norm_u
                return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180.0 / np.pi)
        return 0.0

    def _get_wrist_speed(self, obj):
        """Tốc độ cổ tay lớn nhất (keypoint 9,10) theo chiều cao người, đơn vị frame."""
        if len(obj.pose_history) < 2:
            return 0.0
        speeds = []
        h = max(obj.bbox_history[-1][3] - obj.bbox_history[-1][1], 1e-5)
        for i in range(1, len(obj.pose_history)):
            prev_p, curr_p = obj.pose_history[i - 1], obj.pose_history[i]
            for idx in [9, 10]:  # Wrists
                if prev_p[idx][2] > 0.3 and curr_p[idx][2] > 0.3:
                    disp = np.linalg.norm(curr_p[idx][:2] - prev_p[idx][:2])
                    speeds.append(disp / (h * self.dt))
        return np.max(speeds) if len(speeds) > 0 else 0.0