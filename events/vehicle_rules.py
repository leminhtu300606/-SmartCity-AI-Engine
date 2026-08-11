import numpy as np
import config


class VehicleAccidentRules:
    """Phát hiện va chạm xe-xe — bỏ bớt đặc trưng thừa, giữ tín hiệu cốt lõi.

    Va chạm = GẦN nhau (proximity) + ĐỘNG HỌC bất thường (closing / decel),
    duy trì >= VEHICLE_COLLISION_SUSTAINED frames. Các đặc trưng dễ nhiễu
    (tilt, dist_drop, direction, post_contact, deform) được loại bỏ để giảm
    false positive.
    """

    def __init__(self):
        self.collision_state = {}     # (min_id, max_id) -> sustained_count
        self.deform_state = {}        # ("deform", track_id) -> sustained_count

    def check_collision(self, objA, objB, other_bboxes=None, frame_size=None):
        """Xác định va chạm: proximity (gate) + closing/decel, sustained."""
        if len(objA.bbox_history) < 3 or len(objB.bbox_history) < 3:
            return False, 0.0

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        pair_key = (min(objA.track_id, objB.track_id),
                    max(objA.track_id, objB.track_id))

        # GATE: bbox chồng nhau hoặc cực gần (theo đường chéo trung bình)
        diagA = np.sqrt((boxA[2] - boxA[0]) ** 2 + (boxA[3] - boxA[1]) ** 2)
        diagB = np.sqrt((boxB[2] - boxB[0]) ** 2 + (boxB[3] - boxB[1]) ** 2)
        avg_diag = max((diagA + diagB) / 2.0, 1e-5)

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        dist = float(np.linalg.norm(centA - centB))
        rel_dist = dist / avg_diag
        iou = self._calculate_iou(boxA, boxB)

        is_proximate = (iou > config.VEHICLE_IOU_THRESH
                        or rel_dist < config.VEHICLE_PROXIMITY_DIST_RATIO)
        if not is_proximate:
            self.collision_state[pair_key] = max(
                0, self.collision_state.get(pair_key, 0) - 1
            )
            return False, 0.0

        # Động học bất thường: tiến lại nhanh HOẶC giảm tốc đột ngột
        closing_speed = self._get_closing_speed(objA, objB)
        decel = max(self._get_deceleration(objA), self._get_deceleration(objB))
        kinetic = max(closing_speed / config.VEHICLE_CLOSING_SPEED_THRESH,
                      decel / config.VEHICLE_DECEL_THRESH)

        is_candidate = kinetic > config.VEHICLE_COLLISION_MIN_KINETIC

        if is_candidate:
            self.collision_state[pair_key] = min(
                config.VEHICLE_COLLISION_SUSTAINED + 2,
                self.collision_state.get(pair_key, 0) + 1,
            )
        else:
            self.collision_state[pair_key] = max(
                0, self.collision_state.get(pair_key, 0) - 1
            )

        is_confirmed = (is_candidate
                        and self.collision_state.get(pair_key, 0)
                        >= config.VEHICLE_COLLISION_SUSTAINED)
        score = min(1.0, 0.6 + kinetic * 0.4)
        return is_confirmed, score

    def check_deformation(self, obj, other_bboxes=None, frame_size=None):
        """Xe ĐƠN bị biến dạng (aspect ratio lệch baseline) = tai nạn.

        Loại trừ trường hợp bị màn hình cắt (bbox chạm frame edge).
        """
        if len(obj.bbox_history) < 6:
            return False, 0.0

        box = obj.bbox_history[-1]
        if frame_size is not None:
            fw, fh = frame_size
            margin = config.VEHICLE_DEFORM_OCL_FRAME_MARGIN
            if (box[0] <= margin or box[1] <= margin
                    or box[2] >= fw - margin or box[3] >= fh - margin):
                return False, 0.0

        window = min(12, len(obj.bbox_history))
        aspects = []
        for b in list(obj.bbox_history)[-window:]:
            w = b[2] - b[0]
            h = b[3] - b[1]
            if h < 1e-5 or w < 1e-5:
                continue
            aspects.append(w / h)
        if len(aspects) < 4:
            return False, 0.0

        half = max(1, len(aspects) // 2)
        baseline = float(np.median(aspects[:half]))
        cur_aspect = aspects[-1]
        if baseline < 1e-5:
            return False, 0.0
        deformation = abs(cur_aspect - baseline) / baseline

        key = ("deform", obj.track_id)
        is_candidate = deformation >= config.VEHICLE_DEFORM_MIN_SCORE
        if is_candidate:
            self.deform_state[key] = min(
                config.VEHICLE_COLLISION_SUSTAINED + 2,
                self.deform_state.get(key, 0) + 1,
            )
        else:
            self.deform_state[key] = max(
                0, self.deform_state.get(key, 0) - 1
            )

        is_confirmed = (is_candidate
                        and self.deform_state.get(key, 0)
                        >= config.VEHICLE_COLLISION_SUSTAINED)
        return is_confirmed, deformation

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _get_closing_speed(self, objA, objB):
        """Tốc độ 2 xe tiến lại gần nhau (chiếu relative velocity lên đường nối tâm)."""
        if len(objA.velocity_history) == 0 or len(objB.velocity_history) == 0:
            return 0.0

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        d = centB - centA
        dist = float(np.linalg.norm(d))
        if dist < 1e-5:
            return 0.0
        rel_vel = np.array(objA.velocity_history[-1]) - np.array(objB.velocity_history[-1])
        return max(0.0, float(np.dot(rel_vel, d / dist)))

    def _get_deceleration(self, obj):
        """Giảm tốc đột ngột (px/s²) — chiếu gia tốc ngược hướng vận tốc."""
        if len(obj.accel_history) < 2 or len(obj.velocity_history) == 0:
            return 0.0
        acc = obj.accel_history[-1]
        vel = obj.velocity_history[-1]
        speed = np.linalg.norm(vel)
        if speed < 1e-3:
            return 0.0
        return max(0.0, -float(np.dot(acc, vel)) / speed)

    @staticmethod
    def _calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def cleanup_lost_tracks(self, active_track_ids):
        """Xoá state cho các track đã mất dấu."""
        lost_pairs = [k for k in self.collision_state
                      if k[0] not in active_track_ids or k[1] not in active_track_ids]
        for k in lost_pairs:
            del self.collision_state[k]

        lost_deform = [k for k in self.deform_state
                       if k[1] not in active_track_ids]
        for k in lost_deform:
            del self.deform_state[k]