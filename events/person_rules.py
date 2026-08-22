import numpy as np
import config
from events.scores import fall_score, fight_score


class PersonActionRules:
    """Ngã (fall) & xô xát (fight) — EMIT SCORED CANDIDATE (Rule 6).

    Rule 6A (FALL): không chỉ dựa "person + bbox thay đổi".
      FallScore = posture/vertical/aspect/center_velocity/temporal >= 0.80
      + >= 3/5 frames mới CONFIRMED (do confirm tracker xử lý).
    Rule 6B (FIGHT): 2 người gần không được là điều kiện đủ — phải có
      contact + relative motion mạnh + duy trì.
    """

    def __init__(self):
        self.conflict_state = {}  # (min_id, max_id) -> sustained_count

    # ------------------------------------------------------------
    # FALL — eval trên detection frame
    # ------------------------------------------------------------
    def eval_fall(self, obj):
        """Trả candidate {event_type, confidence=FallScore,...} hoặc None."""
        if len(obj.bbox_history) < 2:
            return None

        curr_bbox = obj.bbox_history[-1]
        w = curr_bbox[2] - curr_bbox[0]
        h = max(curr_bbox[3] - curr_bbox[1], 1e-5)
        is_horizontal = (w / h) > config.FALL_ASPECT_RATIO_THRESH

        if is_horizontal:
            obj.fall_persist_count += 1
        else:
            obj.fall_persist_count = max(0, obj.fall_persist_count - 1)

        comps, score = fall_score(obj)
        if score < config.FALL_CANDIDATE_THRESH:
            return None

        return {
            "event_type": "HUMAN_FALL",
            "track_ids": [obj.track_id],
            "bbox": [int(v) for v in curr_bbox],
            "confidence": round(score, 3),
            "score_components": comps,
            "description": "Phát hiện người bị ngã (FallScore %.2f)" % score,
            "evidence_objects": [self._evidence(obj)],
        }

    # ------------------------------------------------------------
    # FIGHT — eval trên detection frame (cặp người)
    # ------------------------------------------------------------
    def eval_conflict(self, objA, objB, tool_objects=None):
        """Trả candidate xô xát cho cặp (A,B) hoặc None.

        Khoảng cách gần là GATE (không phải điều kiện đủ). FightScore phải
        cao do contact + motion + temporal.
        """
        if len(objA.bbox_history) < 3 or len(objB.bbox_history) < 3:
            return None

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        avg_h = max(((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2.0, 1e-5)
        dist = np.linalg.norm(centA - centB) / avg_h

        pair_key = (min(objA.track_id, objB.track_id),
                    max(objA.track_id, objB.track_id))

        # GATE: quá xa -> không tương tác, decay count
        if dist > config.CONFLICT_DIST_HARD_CAP:
            self.conflict_state[pair_key] = max(
                0, self.conflict_state.get(pair_key, 0) - 1)
            return None

        # Vận động bất thường (jitter / body speed)
        jitter = max(self._get_bbox_jitter(objA), self._get_bbox_jitter(objB))
        body_speed = max(self._get_body_speed(objA), self._get_body_speed(objB))
        agitation = max(jitter, body_speed * 2.0)

        is_candidate = agitation > config.CONFLICT_AGITATION_THRESH
        if is_candidate:
            self.conflict_state[pair_key] = min(
                config.CONFLICT_SUSTAINED_FRAMES + 2,
                self.conflict_state.get(pair_key, 0) + 1)
        else:
            self.conflict_state[pair_key] = max(
                0, self.conflict_state.get(pair_key, 0) - 1)

        sustained = self.conflict_state.get(pair_key, 0)
        comps, score = fight_score(objA, objB, agitation, sustained)
        if score < config.FIGHT_CANDIDATE_THRESH:
            return None

        return {
            "event_type": "HUMAN_CONFLICT",
            "track_ids": [objA.track_id, objB.track_id],
            "bbox": self._union_bbox(objA, objB),
            "confidence": round(score, 3),
            "score_components": comps,
            "description":
                "Phát hiện xô xát/đánh nhau (FightScore %.2f)" % score,
            "zone_name": f"grid_{config.grid_zone(centA[0], centA[1])}",
            "evidence_objects": [self._evidence(objA), self._evidence(objB)],
        }

    def cleanup_lost_tracks(self, active_track_ids):
        """Xoá conflict state cho các track đã mất dấu."""
        lost_pairs = [k for k in self.conflict_state
                      if k[0] not in active_track_ids
                      or k[1] not in active_track_ids]
        for k in lost_pairs:
            del self.conflict_state[k]

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _get_bbox_jitter(self, obj):
        if len(obj.center_history) < 5:
            return 0.0
        window = min(8, len(obj.center_history))
        centers = np.array(list(obj.center_history)[-window:])
        deltas = np.diff(centers, axis=0)
        magnitudes = np.linalg.norm(deltas, axis=1)
        significant_mask = magnitudes > 2.0
        if np.sum(significant_mask) < 3:
            return 0.0
        speed_std = float(np.std(magnitudes[significant_mask]))
        sig_deltas = deltas[significant_mask]
        reversals = 0
        for dim in range(2):
            signs = np.sign(sig_deltas[:, dim])
            reversals += int(np.sum(np.abs(np.diff(signs)) > 0))
        return speed_std + reversals * 2.0

    def _get_body_speed(self, obj):
        if not obj.velocity_history or not obj.bbox_history:
            return 0.0
        box = obj.bbox_history[-1]
        h = max(box[3] - box[1], 1e-5)
        return float(np.linalg.norm(obj.velocity_history[-1])) / h

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