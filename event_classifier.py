import numpy as np
import cv2
import config

class RuleBasedEventClassifier:
    def __init__(self, fps=25):
        self.fps = fps
        self.dt = 1.0 / fps

    def evaluate_events(self, camera_id, memory_manager):
        detected_events = []
        objects = memory_manager.objects

        # 1. Check Person Fall & Intrusion
        for t_id, obj in objects.items():
            if obj.cls_id == 0:  # Person
                # Check Intrusion
                if self._check_intrusion(camera_id, obj):
                    detected_events.append({
                        "event_type": "ZONE_INTRUSION",
                        "track_ids": [int(t_id)],
                        "confidence": 0.95,
                        "description": "Xâm nhập khu vực cấm"
                    })

                # Check Fall
                if self._check_fall(obj):
                    detected_events.append({
                        "event_type": "HUMAN_FALL",
                        "track_ids": [int(t_id)],
                        "confidence": 0.90,
                        "description": "Phát hiện người bị ngã"
                    })

        # 2. Check Conflict giữa các cặp Person
        person_ids = [t_id for t_id, obj in objects.items() if obj.cls_id == 0]
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                idA, idB = person_ids[i], person_ids[j]
                is_fight, score = self._check_conflict(objects[idA], objects[idB])
                if is_fight:
                    detected_events.append({
                        "event_type": "HUMAN_CONFLICT",
                        "track_ids": [int(idA), int(idB)],
                        "confidence": float(score),
                        "description": f"Phát hiện xô xát/đánh nhau (Score: {score:.2f})"
                    })

        # 3. Check Vehicle Accident
        vehicle_ids = [t_id for t_id, obj in objects.items() if obj.cls_id in [2, 3, 5, 7]] # Car, Motorbike, Bus, Truck
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                idA, idB = vehicle_ids[i], vehicle_ids[j]
                if self._check_vehicle_collision(objects[idA], objects[idB]):
                    detected_events.append({
                        "event_type": "VEHICLE_ACCIDENT",
                        "track_ids": [int(idA), int(idB)],
                        "confidence": 0.88,
                        "description": "Xảy ra va chạm giao thông giữa các phương tiện"
                    })

        return detected_events

    def _check_intrusion(self, camera_id, obj):
        if camera_id not in config.RESTRICTED_ROIS:
            return False
        roi_pts = np.array(config.RESTRICTED_ROIS[camera_id], np.int32)
        curr_center = obj.center_history[-1]
        
        is_inside = cv2.pointPolygonTest(roi_pts, (float(curr_center[0]), float(curr_center[1])), False) >= 0
        if is_inside:
            obj.dwell_time += 1
        else:
            obj.dwell_time = max(0, obj.dwell_time - 1)
            
        return obj.dwell_time >= config.INTRUSION_DWELL_FRAMES

    def _check_fall(self, obj):
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

        theta_torso = 0.0
        if len(obj.pose_history) > 0:
            curr_pose = obj.pose_history[-1]
            # Keypoint 5,6: Vai | Keypoint 11,12: Hông
            if curr_pose[5][2] > 0.3 and curr_pose[6][2] > 0.3 and curr_pose[11][2] > 0.3 and curr_pose[12][2] > 0.3:
                neck = (curr_pose[5][:2] + curr_pose[6][:2]) / 2.0
                hip = (curr_pose[11][:2] + curr_pose[12][:2]) / 2.0
                u = hip - neck
                norm_u = np.linalg.norm(u)
                if norm_u > 0:
                    cos_theta = u[1] / norm_u
                    theta_torso = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180.0 / np.pi)

        cond_drop = max_v_y > config.FALL_V_Y_THRESH
        cond_lying = (aspect_ratio > config.FALL_ASPECT_RATIO_THRESH) or (theta_torso > config.FALL_TORSO_ANGLE_THRESH)
        return cond_drop and cond_lying

    def _check_conflict(self, objA, objB):
        if len(objA.bbox_history) < 10 or len(objB.bbox_history) < 10:
            return False, 0.0

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        centA, centB = np.array(objA.center_history[-1]), np.array(objB.center_history[-1])
        avg_h = ((boxA[3]-boxA[1]) + (boxB[3]-boxB[1])) / 2.0
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

    def _get_wrist_speed(self, obj):
        if len(obj.pose_history) < 2:
            return 0.0
        speeds = []
        h = max(obj.bbox_history[-1][3] - obj.bbox_history[-1][1], 1e-5)
        for i in range(1, len(obj.pose_history)):
            prev_p, curr_p = obj.pose_history[i-1], obj.pose_history[i]
            for idx in [9, 10]: # Wrists
                if prev_p[idx][2] > 0.3 and curr_p[idx][2] > 0.3:
                    disp = np.linalg.norm(curr_p[idx][:2] - prev_p[idx][:2])
                    speeds.append(disp / (h * self.dt))
        return np.max(speeds) if len(speeds) > 0 else 0.0

    def _check_vehicle_collision(self, objA, objB):
        if len(objA.velocity_history) < 5 or len(objB.velocity_history) < 5:
            return False

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        inter_w = max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0]))
        inter_h = max(0, min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]))
        if inter_w * inter_h <= 0:
            return False

        # Kiểm tra sự giảm tốc đột ngột (Sudden Deceleration)
        vA = [np.linalg.norm(v) for v in objA.velocity_history]
        accA = np.diff(vA) / self.dt
        has_sudden_stop = np.min(accA) < config.ACCIDENT_DECEL_THRESH if len(accA) > 0 else False

        return has_sudden_stop