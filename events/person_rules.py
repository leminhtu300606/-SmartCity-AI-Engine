import numpy as np
import config


class PersonActionRules:
    """Xử lý ngã, giằng co, xô xát dựa trên Temporal History & Pose.

    Cải tiến:
    - Normalize velocity/acceleration theo chiều cao người (không dùng pixel tuyệt đối).
    - Fall persistence: phải ở tư thế nằm ngang >= N frame mới confirm.
    - BBox-based jitter fallback: khi pose unavailable, dùng center oscillation để detect conflict.
    - Conflict sustained: xung đột phải duy trì >= N frame.
    """

    def __init__(self):
        # Temporal state cho sustained pairwise detection
        self.conflict_state = {}  # (min_id, max_id) -> sustained_count
        self.collision_state = {}  # (min_id, max_id) -> sustained_count

    def check_fall(self, obj):
        """Phát hiện người ngã qua Aspect Ratio, Torso Angle & Gia tốc rơi.

        Thresholds được chuẩn hóa theo chiều cao người để không phụ thuộc resolution.
        Phải duy trì tư thế ngã >= FALL_PERSIST_FRAMES detection frames.
        """
        if len(obj.bbox_history) < 5:
            return False

        curr_bbox = obj.bbox_history[-1]
        w = curr_bbox[2] - curr_bbox[0]
        h = max(curr_bbox[3] - curr_bbox[1], 1e-5)
        aspect_ratio = w / h
        person_height = h

        # Torso Angle từ Pose (Keypoint 5,6: Shoulders; 11,12: Hips)
        theta_torso = self._get_torso_angle(obj)

        # Normalized velocity & acceleration (chia cho chiều cao người)
        vel_y_norm = [v[1] / max(person_height, 1.0)
                      for v in list(obj.velocity_history)[-5:]]
        acc_y_norm = [a[1] / max(person_height, 1.0)
                      for a in list(obj.accel_history)[-5:]]
        max_vel_y = max(vel_y_norm, default=0.0)

        is_horizontal = (aspect_ratio > config.FALL_ASPECT_RATIO_THRESH
                         or theta_torso > config.FALL_TORSO_ANGLE_THRESH)
        is_falling_motion = (max_vel_y > config.FALL_VEL_NORM_THRESH
                             or any(a > config.FALL_ACCEL_NORM_THRESH for a in acc_y_norm))

        # Temporal aspect ratio check: phải nằm ngang nhiều frame liên tiếp
        recent_aspects = []
        for box in list(obj.bbox_history)[-5:]:
            bw = box[2] - box[0]
            bh = max(box[3] - box[1], 1e-5)
            recent_aspects.append(bw / bh)
        is_sustained_horizontal = (
            sum(1 for a in recent_aspects
                if a > config.FALL_ASPECT_RATIO_THRESH * 0.9) >= 3
        )

        is_candidate = is_horizontal and (is_falling_motion or is_sustained_horizontal)

        # Update persistence counter trên object
        if is_candidate:
            obj.fall_persist_count += 1
        else:
            obj.fall_persist_count = max(0, obj.fall_persist_count - 1)

        return obj.fall_persist_count >= config.FALL_PERSIST_FRAMES

    def check_conflict(self, objA, objB):
        """Phát hiện 2 người xô xát/giằng co/đánh nhau — bộ quyết định 2 đường.

        KHÔNG coi "nhiều người trong khung hình" là đánh nhau. Phải có HÀNH VI
        thật sự:
          - Đám đông đứng nói chuyện / đi chung: gần nhau NHƯNG không ai vung
            tay nhanh và không ai cử động giật cục mạnh → loại.
          - Một người chạy ngang / vẫy tay gần người khác: chỉ 1 người động,
            người kia không phản ứng, khoảng cách ổn định → loại.

        ĐƯỜNG A (Pose): 1 người vung cổ tay RẤT nhanh (đấm/đánh) + gần nhau +
        (người kia cũng động HOẶC khoảng cách dao động).
        ĐƯỜNG B (BBox fallback): CẢ HAI đều cử động giật cục mạnh VÀ khoảng
        cách dao động mạnh (giằng co lúc gần lúc xa).
        """
        if len(objA.bbox_history) < 5 or len(objB.bbox_history) < 5:
            return False, 0.0

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        avg_h = max(((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2.0, 1e-5)
        dist = np.linalg.norm(centA - centB) / avg_h

        pair_key = (min(objA.track_id, objB.track_id),
                    max(objA.track_id, objB.track_id))

        # ============================================================
        # GATE: gần nhau (proximity)
        # Người đi chung đường cũng có thể gần nhau → chưa đủ, chỉ là gate.
        # ============================================================
        if dist > config.CONFLICT_DIST_THRESH:
            self.conflict_state[pair_key] = max(
                0, self.conflict_state.get(pair_key, 0) - 1
            )
            return False, 0.0

        # ============================================================
        # ĐẶC TRƯNG: cử động của TỪNG người
        # ============================================================
        wrist_A = self._get_wrist_speed(objA)
        wrist_B = self._get_wrist_speed(objB)
        jitter_A = self._get_bbox_jitter(objA)
        jitter_B = self._get_bbox_jitter(objB)

        # Độ kích động của mỗi người = max(wrist speed, bbox jitter)
        agitation_A = max(wrist_A, jitter_A)
        agitation_B = max(wrist_B, jitter_B)
        mutual_agitation = min(agitation_A, agitation_B)

        # ============================================================
        # ĐẶC TRƯNG: khoảng cách dao động (giằng co)
        # Trong đánh nhau, 2 người lúc áp sát lúc tách xa → phương sai khoảng
        # cách lớn. Nói chuyện/đứng yên/đi chung → khoảng cách ổn định.
        # ============================================================
        window = min(8, min(len(objA.center_history), len(objB.center_history)))
        dists = [
            np.linalg.norm(np.array(cA) - np.array(cB)) / avg_h
            for cA, cB in zip(
                list(objA.center_history)[-window:],
                list(objB.center_history)[-window:],
            )
        ]
        dist_variance = np.var(dists) if len(dists) > 1 else 0.0

        # Bbox overlap (IoU): vật lộn/giằng co → 2 người dính sát, bbox chồng
        # nhau nhiều. Ôm nhau / nói chuyện sát cũng có IoU cao NHƯNG jitter=0.
        bbox_iou = self._calculate_iou(boxA, boxB)

        # ============================================================
        # QUYẾT ĐỊNH 3 ĐƯỜNG
        # ============================================================
        # Đường A — Pose: có người vung cổ tay nhanh (đấm/đánh)
        pose_punch = (max(wrist_A, wrist_B) > config.CONFLICT_WRIST_HIGH_THRESH)

        # Đường B — BBox: CẢ HAI người cử động giật cục rất mạnh.
        # Jitter cao (>= 12) chỉ đạt khi thực sự giằng co/đánh nhau; người đi
        # lại bình thường chỉ ~7. Giằng co sát nhau có khoảng cách ổn định
        # nên không dùng oscillation làm điều kiện bắt buộc.
        both_jittery = (min(jitter_A, jitter_B)
                        > config.CONFLICT_BBOX_JITTER_THRESH)

        # Đường C — Grapple (vật lộn tại chỗ): 2 người dính sát, bbox chồng
        # nhau (IoU cao) + ít nhất 1 người giằng co đáng kể. Đây là kiểu xô
        # xát "bế/ôm/quật" mà 1 người bị khống chế có thể gần như đứng yên
        # (jitter thấp) — không đạt signal_mutual của đường A/B.
        # Phân biệt:
        #   - Ôm nhau/nói chuyện: cả 2 gần như đứng yên (jitter ≈ 0) → không
        #     đạt grapple_activity.
        #   - Đi ngang/đứng gần: 2 bbox không chồng nhiều (IoU thấp) → không
        #     đạt grapple_overlap.
        #   - Đám đông chen lấn thoáng qua: IoU thấp (bbox cạnh nhau).
        # IoU_THRESH để cao để chỉ bắt khi 2 người thực sự CHỒNG VÀO nhau.
        grapple_overlap = (bbox_iou > config.CONFLICT_GRAPPLE_IOU_THRESH
                           and dist < config.CONFLICT_GRAPPLE_DIST_THRESH)
        grapple_activity = (max(jitter_A, jitter_B)
                            > config.CONFLICT_GRAPPLE_MIN_JITTER)

        # Tương tác: người kia cũng động hoặc khoảng cách dao động mạnh
        signal_mutual = mutual_agitation > config.CONFLICT_MUTUAL_AGITATION_THRESH
        signal_oscillation = dist_variance > config.CONFLICT_DIST_VAR_THRESH

        # Đường A: vung tay + có tương tác
        path_pose = pose_punch and (signal_mutual or signal_oscillation)

        # Đường B: cả 2 giật mạnh + có tương tác (đôi công / giằng co sát nhau)
        path_bbox = both_jittery and (signal_mutual or signal_oscillation)

        # Đường C: vật lộn — 2 người CHỒNG bbox + ít nhất 1 người giằng co.
        # Không cần signal_mutual/oscillation vì "người bị giữ" thường đứng
        # yên (đó là bản chất của vật lộn: 1 người khống chế, 1 người giãy).
        # Ôm nhau: grapple_activity fail (jitter≈0). Đi ngang: grapple_overlap
        # fail (IoU thấp).
        path_grapple = (grapple_overlap and grapple_activity)

        is_candidate = path_pose or path_bbox or path_grapple

        # Điểm kết hợp (chỉ để tính confidence)
        conflict_score = (max(wrist_A, wrist_B) * 0.4
                          + (jitter_A + jitter_B) * 0.2
                          + dist_variance * 40.0 * 0.3
                          + mutual_agitation * 0.1
                          + bbox_iou * 20.0 * 0.2)

        # Temporal sustained confirmation
        # Cap counter tại SUSTAINED + 2: khi hiện tượng dừng, chỉ cần decay
        # 2-3 frame là dưới ngưỡng → cảnh báo tắt ngay, không treo lâu.
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
        return is_confirmed, conflict_score

    def check_person_collision(self, objA, objB):
        """2 người va chạm / đụng nhau — bộ quyết định ĐA ĐƯỜNG.

        Va chạm giữa người thường rất nhanh nên các tín hiệu riêng lẻ có thể
        bị nhiễu 1 frame. Gộp 3 đường:
          ĐƯỜNG A — Tiếp cận nhanh: 2 người lao vào nhau (closing speed cao).
          ĐƯỜNG B — Di chuyển bất thường: ít nhất 1 người di chuyển nhanh lạ
          thường (speed vượt đi bộ/chạy thường), gia tốc/giảm tốc đột ngột,
          hoặc đổi hướng đột ngột — dấu hiệu chạy xô vào nhau / phanh gấp.
          ĐƯỜNG C — Đẩy ngã: 1 người ở tư thế ngã (nằm ngang / rơi nhanh)
          khi 2 người rất gần nhau — dấu hiệu bị va/đẩy ngã.
        Tất cả cần proximity (dist) làm gate + duy trì >= N frame.
        """
        if len(objA.velocity_history) < 2 or len(objB.velocity_history) < 2:
            return False, 0.0

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        avg_h = max(((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2.0, 1e-5)
        dist = np.linalg.norm(centA - centB) / avg_h

        pair_key = (min(objA.track_id, objB.track_id),
                    max(objA.track_id, objB.track_id))

        if dist > config.PERSON_APPROACH_DIST_THRESH:
            self.collision_state[pair_key] = max(
                0, self.collision_state.get(pair_key, 0) - 1
            )
            return False, 0.0

        # ============================================================
        # ĐƯỜNG A — Closing speed (tiếp cận nhanh), trung bình vài frame
        # ============================================================
        closing_speeds = []
        window = min(3, min(len(objA.center_history), len(objB.center_history)) - 1)
        for i in range(window):
            idx = -(i + 1)
            cA = np.array(objA.center_history[idx])
            cB = np.array(objB.center_history[idx])
            diff = cB - cA
            norm = np.linalg.norm(diff)
            if norm < 1e-5:
                continue
            dir_vec = diff / norm
            vA = np.array(objA.velocity_history[idx])
            vB = np.array(objB.velocity_history[idx])
            closing_speeds.append(float(np.dot(vA - vB, dir_vec)))

        avg_closing = np.mean(closing_speeds) if closing_speeds else 0.0
        fast_approach = avg_closing > config.PERSON_APPROACH_SPEED_THRESH

        # ============================================================
        # ĐƯỜNG B — Di chuyển bất thường (nhanh / đột ngột / đổi hướng)
        # ============================================================
        speedA = float(np.linalg.norm(np.array(objA.velocity_history[-1])))
        speedB = float(np.linalg.norm(np.array(objB.velocity_history[-1])))
        max_speed = max(speedA, speedB)
        abnormal_speed = max_speed > config.PERSON_ABNORMAL_SPEED_THRESH

        accelA = float(np.linalg.norm(np.array(objA.accel_history[-1])))
        accelB = float(np.linalg.norm(np.array(objB.accel_history[-1])))
        max_accel = max(accelA, accelB)
        sudden_accel = max_accel > config.PERSON_ABNORMAL_ACCEL_THRESH

        dir_change = 0.0
        if (len(objA.direction_history) >= 2 and len(objB.direction_history) >= 2):
            dA = objA.direction_history[-1] - objA.direction_history[-2]
            dB = objB.direction_history[-1] - objB.direction_history[-2]
            # Wrap to [-pi, pi]
            dA = abs((dA + np.pi) % (2 * np.pi) - np.pi)
            dB = abs((dB + np.pi) % (2 * np.pi) - np.pi)
            dir_change = max(dA, dB)
        sudden_turn = dir_change > config.PERSON_DIR_CHANGE_THRESH

        abnormal_motion = (abnormal_speed or sudden_accel or sudden_turn)

        # ============================================================
        # ĐƯỜNG C — Đẩy ngã: 1 người ngã khi 2 người rất gần nhau
        # ============================================================
        fall_prox = dist < config.PERSON_FALL_PROX_DIST
        fell = self._is_falling(objA) or self._is_falling(objB)
        push_fall = fall_prox and fell

        # ============================================================
        # KẾT HỢP + SUSTAINED
        # Va chạm thật = 1 người chuyển động BẤT THƯỜNG (chạy nhanh/đột ngột/
        # đổi hướng) khi 2 người đang rất gần hoặc đang lao vào nhau (closing
        # cao). Hai người đi ngang/cạnh nhau với tốc độ bình thường (speed
        # không bất thường) → không báo. Đẩy ngã bắt riêng.
        # ============================================================
        close_prox = dist < 0.35
        is_candidate = (
            push_fall
            or (abnormal_motion and (close_prox or fast_approach))
        )

        # Điểm va chạm (confidence) — kết hợp các tín hiệu
        collision_score = (
            min(avg_closing / 150.0, 1.0) * 0.30
            + min(max_speed / 120.0, 1.0) * 0.25
            + min(max_accel / 200.0, 1.0) * 0.20
            + min(dir_change / 3.0, 1.0) * 0.15
            + (1.0 if push_fall else 0.0) * 0.10
        )

        if is_candidate:
            self.collision_state[pair_key] = min(
                config.PERSON_COLLISION_SUSTAINED + 2,
                self.collision_state.get(pair_key, 0) + 1,
            )
        else:
            self.collision_state[pair_key] = max(
                0, self.collision_state.get(pair_key, 0) - 1
            )

        is_confirmed = (is_candidate
                        and self.collision_state.get(pair_key, 0)
                        >= config.PERSON_COLLISION_SUSTAINED)
        return is_confirmed, collision_score

    def _is_falling(self, obj):
        """1 người đang ở tư thế ngã: nằm ngang hoặc rơi nhanh theo trục dọc."""
        if len(obj.bbox_history) < 2:
            return False
        box = obj.bbox_history[-1]
        w = box[2] - box[0]
        h = max(box[3] - box[1], 1e-5)
        aspect_ratio = w / h
        person_height = h

        vel_y_norm = [v[1] / max(person_height, 1.0)
                      for v in list(obj.velocity_history)[-3:]]
        max_vel_y = max(vel_y_norm, default=0.0)
        acc_y_norm = [a[1] / max(person_height, 1.0)
                      for a in list(obj.accel_history)[-3:]]
        falling_motion = (max_vel_y > config.FALL_VEL_NORM_THRESH
                          or any(a > config.FALL_ACCEL_NORM_THRESH for a in acc_y_norm))
        return (aspect_ratio > config.FALL_ASPECT_RATIO_THRESH
                or falling_motion)

    def cleanup_lost_tracks(self, active_track_ids):
        """Xoá conflict/collision state cho các track đã mất dấu."""
        for state_map in (self.conflict_state, self.collision_state):
            lost_pairs = [k for k in state_map
                          if k[0] not in active_track_ids
                          or k[1] not in active_track_ids]
            for k in lost_pairs:
                del state_map[k]

    # ----------------------------------------------------------------
    # Private Helpers
    # ----------------------------------------------------------------

    def _calculate_iou(self, boxA, boxB):
        """Bbox IoU giữa 2 người — dùng cho đường vật lộn (grapple)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def _get_bbox_jitter(self, obj):
        """Agitation score từ movement pattern analysis (fallback khi không có pose).

        Đo biên độ dao động + số lần đổi hướng của center object.
        Người đang đánh nhau có chuyển động giật cục, đổi hướng liên tục.
        """
        if len(obj.center_history) < 5:
            return 0.0

        window = min(8, len(obj.center_history))
        centers = np.array(list(obj.center_history)[-window:])

        # Frame-to-frame displacement vectors
        deltas = np.diff(centers, axis=0)
        magnitudes = np.linalg.norm(deltas, axis=1)

        # Cần chuyển động đáng kể
        significant_mask = magnitudes > 2.0
        if np.sum(significant_mask) < 3:
            return 0.0

        # Speed variance = tốc độ thay đổi thất thường
        speed_std = float(np.std(magnitudes[significant_mask]))

        # Direction reversals = số lần đổi hướng (sign changes trên dx, dy)
        sig_deltas = deltas[significant_mask]
        reversals = 0
        for dim in range(2):
            signs = np.sign(sig_deltas[:, dim])
            reversals += int(np.sum(np.abs(np.diff(signs)) > 0))

        return speed_std + reversals * 2.0

    def _get_torso_angle(self, obj):
        if len(obj.pose_history) == 0 or obj.pose_history[-1] is None:
            return 0.0
        kp = obj.pose_history[-1]  # Shape: (17, 3) [x, y, conf]
        if len(kp) <= 12:
            return 0.0
        if kp[5][2] > 0.3 and kp[6][2] > 0.3 and kp[11][2] > 0.3 and kp[12][2] > 0.3:
            neck = np.asarray(kp[5][:2], dtype=float) + np.asarray(kp[6][:2], dtype=float)
            hip = np.asarray(kp[11][:2], dtype=float) + np.asarray(kp[12][:2], dtype=float)
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
        # Wrist Indices: 9 (Left), 10 (Right) - bỏ qua keypoint thiếu
        speeds = []
        for idx in [9, 10]:
            if (idx < len(kp_curr) and idx < len(kp_prev)
                    and kp_curr[idx][2] > 0.3 and kp_prev[idx][2] > 0.3):
                cur = np.asarray(kp_curr[idx][:2], dtype=float)
                prev = np.asarray(kp_prev[idx][:2], dtype=float)
                spd = float(np.linalg.norm(cur - prev) / dt)
                speeds.append(spd)
        return max(speeds, default=0.0)
