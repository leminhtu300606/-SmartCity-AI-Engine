import numpy as np
import config
from events.scores import collision_score


class VehicleAccidentRules:
    """Va chạm xe-xe — EMIT SCORED CANDIDATE (Rule 6C).

    Rule 6C: 2 xe cần track ổn định + tín hiệu động học thật.
    QUAN TRỌNG: nếu track xe có confidence (YOLO) thấp < VEHICLE_COLLISION_MIN_CONF
    → KHÔNG được vào collision pipeline (loại "0.38 car = plastic object").
    """

    def __init__(self):
        self.collision_state = {}     # (min_id, max_id) -> sustained_count
        self.deform_state = {}        # ("deform", track_id) -> sustained_count

    def _gate_ok(self, obj_obj):
        """Vehicle conf gate (Rule 6C): conf >= VEHICLE_COLLISION_MIN_CONF."""
        obj, *_ = obj_obj if isinstance(obj_obj, tuple) else (obj_obj,)
        return getattr(obj, "conf", 0.0) >= config.VEHICLE_COLLISION_MIN_CONF

    def _both_gate_ok(self, objA, objB):
        return (getattr(objA, "conf", 0.0) >= config.VEHICLE_COLLISION_MIN_CONF
                and getattr(objB, "conf", 0.0) >= config.VEHICLE_COLLISION_MIN_CONF)

    def eval_collision(self, objA, objB, other_bboxes=None, frame_size=None):
        """Trả candidate va chạm cho cặp (A,B) hoặc None."""
        if not self._both_gate_ok(objA, objB):
            return None
        if len(objA.bbox_history) < 3 or len(objB.bbox_history) < 3:
            return None

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        pair_key = (min(objA.track_id, objB.track_id),
                    max(objA.track_id, objB.track_id))

        # GATE: bbox chồng nhau hoặc cực gần
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
                0, self.collision_state.get(pair_key, 0) - 1)
            return None

        # Động học: closing / decel
        closing_speed = self._get_closing_speed(objA, objB)
        decel = max(self._get_deceleration(objA), self._get_deceleration(objB))
        kinetic = max(closing_speed / config.VEHICLE_CLOSING_SPEED_THRESH,
                      decel / config.VEHICLE_DECEL_THRESH)

        is_candidate = kinetic > config.VEHICLE_COLLISION_MIN_KINETIC
        if is_candidate:
            self.collision_state[pair_key] = min(
                config.VEHICLE_COLLISION_SUSTAINED + 2,
                self.collision_state.get(pair_key, 0) + 1)
        else:
            self.collision_state[pair_key] = max(
                0, self.collision_state.get(pair_key, 0) - 1)

        sustained = self.collision_state.get(pair_key, 0)
        comps, score = collision_score(
            objA, objB, closing_speed, decel, sustained, track_frames=config.COLLISION_CONFIRM_FRAMES)
        if score < config.COLLISION_CANDIDATE_THRESH:
            return None

        return {
            "event_type": "VEHICLE_COLLISION",
            "track_ids": [objA.track_id, objB.track_id],
            "bbox": self._union_bbox(objA, objB),
            "confidence": score,
            "score_components": comps,
            "description":
                "[VA CHẠM GIAO THÔNG] phương tiện 2 xe va chạm (CollisionScore %.2f)" % score,
            "evidence_objects": [self._evidence(objA), self._evidence(objB)],
        }

    def eval_deformation(self, obj, other_bboxes=None, frame_size=None):
        """Xe ĐƠN bị biến dạng aspect ratio = tai nạn (loại trừ bị frame cắt)."""
        if not self._gate_ok(obj):
            return None
        if len(obj.bbox_history) < 6:
            return None

        box = obj.bbox_history[-1]
        if frame_size is not None:
            fw, fh = frame_size
            margin = config.VEHICLE_DEFORM_OCL_FRAME_MARGIN
            if (box[0] <= margin or box[1] <= margin
                    or box[2] >= fw - margin or box[3] >= fh - margin):
                return None

        window = min(12, len(obj.bbox_history))
        aspects = []
        for b in list(obj.bbox_history)[-window:]:
            w = b[2] - b[0]
            h = b[3] - b[1]
            if h < 1e-5 or w < 1e-5:
                continue
            aspects.append(w / h)
        if len(aspects) < 4:
            return None

        half = max(1, len(aspects) // 2)
        baseline = float(np.median(aspects[:half]))
        cur_aspect = aspects[-1]
        if baseline < 1e-5:
            return None
        deformation = abs(cur_aspect - baseline) / baseline

        key = ("deform", obj.track_id)
        is_candidate = deformation >= config.VEHICLE_DEFORM_MIN_SCORE
        if is_candidate:
            self.deform_state[key] = min(
                config.VEHICLE_COLLISION_SUSTAINED + 2,
                self.deform_state.get(key, 0) + 1)
        else:
            self.deform_state[key] = max(
                0, self.deform_state.get(key, 0) - 1)

        sustained = self.deform_state.get(key, 0)
        if sustained < config.VEHICLE_COLLISION_SUSTAINED:
            is_candidate = False

        if not is_candidate:
            return None

        conf = min(0.93, 0.80 + deformation * 0.15)
        return {
            "event_type": "VEHICLE_COLLISION",
            "track_ids": [obj.track_id],
            "bbox": [int(v) for v in box],
            "confidence": conf,
            "score_components": {"deformation": min(1.0, deformation)},
            "description":
                "[VA CHẠM GIAO THÔNG] phương tiện bị biến dạng đột ngột (deform %.2f)" % deformation,
            "evidence_objects": [self._evidence(obj)],
        }

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _get_closing_speed(self, objA, objB):
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

    @staticmethod
    def _evidence(obj):
        bbox = None
        if len(obj.bbox_history) > 0:
            try:
                bbox = [int(v) for v in obj.bbox_history[-1]]
            except (TypeError, ValueError):
                bbox = None
        speed = 0.0
        if len(obj.velocity_history) > 0:
            vel = obj.velocity_history[-1]
            speed = float((vel[0] ** 2 + vel[1] ** 2) ** 0.5)
        return {
            "track_id": obj.track_id,
            "cls_id": obj.cls_id,
            "bbox": bbox,
            "speed": round(speed, 3),
            "missed_frames": obj.missed_frames,
            "predicted": bool(obj.last_update_predicted),
        }

    @staticmethod
    def _union_bbox(objA, objB):
        bA = [int(v) for v in objA.bbox_history[-1]] if len(objA.bbox_history) else None
        bB = [int(v) for v in objB.bbox_history[-1]] if len(objB.bbox_history) else None
        if bA is None:
            return bB
        if bB is None:
            return bA
        return [
            min(bA[0], bB[0]), min(bA[1], bB[1]),
            max(bA[2], bB[2]), max(bA[3], bB[3]),
        ]

    def cleanup_lost_tracks(self, active_track_ids):
        lost_pairs = [k for k in self.collision_state
                      if k[0] not in active_track_ids or k[1] not in active_track_ids]
        for k in lost_pairs:
            del self.collision_state[k]
        lost_deform = [k for k in self.deform_state if k[1] not in active_track_ids]
        for k in lost_deform:
            del self.deform_state[k]