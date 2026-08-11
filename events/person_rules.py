import numpy as np
import config


class PersonActionRules:
    """Xử lý ngã (fall) và xô xát (conflict) — bỏ bớt điều kiện không cần thiết.

    Giữ các tín hiệu cốt lõi, giảm tham số để tăng độ chính xác:
      - Ngã: người nằm ngang (aspect ratio) duy trì >= FALL_PERSIST_FRAMES.
      - Xô xát: 2 người GẦN nhau + có chuyển động giật cục (jitter) hoặc
        nhanh (body speed), duy trì >= CONFLICT_SUSTAINED_FRAMES.
    """

    def __init__(self):
        self.conflict_state = {}  # (min_id, max_id) -> sustained_count

    def check_fall(self, obj):
        """Ngã = tư thế nằm ngang duy trì đủ số detection frame."""
        if len(obj.bbox_history) < 1:
            return False

        curr_bbox = obj.bbox_history[-1]
        w = curr_bbox[2] - curr_bbox[0]
        h = max(curr_bbox[3] - curr_bbox[1], 1e-5)
        is_horizontal = (w / h) > config.FALL_ASPECT_RATIO_THRESH

        if is_horizontal:
            obj.fall_persist_count += 1
        else:
            obj.fall_persist_count = max(0, obj.fall_persist_count - 1)

        return obj.fall_persist_count >= config.FALL_PERSIST_FRAMES

    def check_conflict(self, objA, objB, tool_objects=None):
        """Xô xát = 2 người GẦN nhau và có vận động bất thường, duy trì N frame.

        Tín hiệu cốt lõi:
          - Khoảng cách 2 người <= CONFLICT_DIST_HARD_CAP (theo avg height).
          - Jitter (giật cục) hoặc body speed (di chuyển nhanh) vượt ngưỡng.
        """
        if len(objA.bbox_history) < 3 or len(objB.bbox_history) < 3:
            return False, 0.0

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        avg_h = max(((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2.0, 1e-5)
        dist = np.linalg.norm(centA - centB) / avg_h

        pair_key = (min(objA.track_id, objB.track_id),
                    max(objA.track_id, objB.track_id))

        # GATE: quá xa -> chắc chắn không tương tác
        if dist > config.CONFLICT_DIST_HARD_CAP:
            self.conflict_state[pair_key] = max(
                0, self.conflict_state.get(pair_key, 0) - 1
            )
            return False, 0.0

        # Vận động bất thường: giật cục hoặc cơ thể di chuyển nhanh
        jitter = max(self._get_bbox_jitter(objA), self._get_bbox_jitter(objB))
        body_speed = max(self._get_body_speed(objA), self._get_body_speed(objB))
        agitation = max(jitter, body_speed * 2.0)

        is_candidate = agitation > config.CONFLICT_AGITATION_THRESH

        # Temporal sustained confirmation
        if is_candidate:
            self.conflict_state[pair_key] = min(
                config.CONFLICT_SUSTAINED_FRAMES + 2,
                self.conflict_state.get(pair_key, 0) + 1,
            )
        else:
            self.conflict_state[pair_key] = max(
                0, self.conflict_state.get(pair_key, 0) - 1
            )

        is_confirmed = (is_candidate
                        and self.conflict_state.get(pair_key, 0)
                        >= config.CONFLICT_SUSTAINED_FRAMES)
        return is_confirmed, agitation

    def cleanup_lost_tracks(self, active_track_ids):
        """Xoá conflict state cho các track đã mất dấu."""
        lost_pairs = [k for k in self.conflict_state
                      if k[0] not in active_track_ids
                      or k[1] not in active_track_ids]
        for k in lost_pairs:
            del self.conflict_state[k]

    # ----------------------------------------------------------------
    # Private Helpers
    # ----------------------------------------------------------------

    def _get_bbox_jitter(self, obj):
        """Chỉ số giật cục từ chuyển động center (bắt xô đẩy, vung tay mạnh)."""
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
        """Tốc độ di chuyển cơ thể (normalized theo chiều cao)."""
        if not obj.velocity_history or not obj.bbox_history:
            return 0.0
        box = obj.bbox_history[-1]
        h = max(box[3] - box[1], 1e-5)
        return float(np.linalg.norm(obj.velocity_history[-1])) / h