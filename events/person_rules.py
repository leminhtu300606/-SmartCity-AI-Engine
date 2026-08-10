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
        self.group_conflict_state = {}  # (id1, id2, id3) -> sustained_count
        self.lying_persist = {}  # track_id -> số frame liên tiếp ở tư thế nằm

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
        """Phát hiện XÔ XÁT / ĐÁNH NHAU — ĐỊNH NGHĨA LẠI theo 3 YẾU TỐ + HẬU QUẢ.

        LÝ THUYẾT:
          XÔ XÁT / ĐÁNH NHAU (gồm: đánh nhau thực tế, xô đẩy/giằng co, đánh võ
          biểu diễn, đánh vật biểu diễn) = tương tác xung đột giữa 2+ người có
          VẬN ĐỘNG THỂ CHẤT DỮ DỘI.
          ÔM NHAU / THÂN NHAU (yêu thương) = tương tác gần gũi nhưng vận động
          CHẬM, không có hành vi tấn công → KHÔNG phải xô xát.

        BÓC TÁCH 3 YẾU TỐ:
          (1) BODY KINETIC — tốc độ di chuyển cơ thể (normalized theo chiều cao)
              hoặc 2 người lao vào nhau nhanh (approach/closing). Đánh nhau =
              nhanh; ôm/thân = chậm hoặc đứng yên.
          (2) AGGRESSION — vung tay nhanh/biên độ lớn HƯỚNG VÀO người kia
              (wrist alignment), đá/gạt chân (ankle), ngã nhanh, nằm vật.
              Võ/vật biểu diễn vẫn có tín hiệu này → vẫn được báo như nhau.
          (3) APPROACH — closing speed (2 người lao vào nhau).
          HẬU QUẢ (≈70% vụ xô xát): 1 bên ngã rồi NẰM LÂU KHÔNG DẬY (sustained
          lying) → bonus mạnh, xác nhận ngay cả khi bắt sót vung tay.

        KẾT LUẬN: (Aggression ∨ Aftermath) ∧ (Kinetic ∨ Aftermath).
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
        # GATE: CÙNG Ô LƯỚI (grid zone)
        # 2 người phải thuộc CÙNG 1 ô lưới trong frame (tâm bbox nằm cùng
        # vùng). Khác ô = 2 khu vực xa nhau → không xét xô xát.
        # ============================================================
        if config.grid_zone(*centA) != config.grid_zone(*centB):
            self.conflict_state[pair_key] = max(
                0, self.conflict_state.get(pair_key, 0) - 1
            )
            return False, 0.0

        # ============================================================
        # STAGE 1: TÌNH TRẠNG của TỪNG người (cá nhân, không phụ thuộc người kia)
        # ============================================================
        condA = self._assess_person_condition(objA)
        condB = self._assess_person_condition(objB)

        # ============================================================
        # YẾU TỐ 1 — BODY KINETIC: cơ thể di chuyển nhanh hoặc lao vào nhau
        # ============================================================
        body_speed = max(condA["body_speed"], condB["body_speed"])
        closing = self._get_closing_speed(objA, objB)  # px/s
        closing_norm = closing / avg_h
        kinetic = (body_speed > config.CONFLICT_BODY_SPEED_THRESH
                   or closing_norm > config.CONFLICT_CLOSING_NORM_THRESH)

        # ============================================================
        # YẾU TỐ 2 — AGGRESSION: vung tay hướng vào người kia / đá / ngã / nằm
        # ============================================================
        wrist_attack = self._get_wrist_attack_component(condA, condB, centA, centB)
        wrist_fast = (max(condA["wrist_speed"], condB["wrist_speed"])
                      > config.CONFLICT_WRIST_HIGH_THRESH)
        wrist_big = (max(condA["wrist_amplitude"], condB["wrist_amplitude"])
                     > config.CONFLICT_WRIST_AMPLITUDE_THRESH)
        leg_kick = (max(condA["leg_speed"], condB["leg_speed"])
                    > config.CONFLICT_LEG_KICK_THRESH)
        fast_fall = condA["falling"] or condB["falling"]
        lying_now = condA["lying"] or condB["lying"]

        aggression = (wrist_attack or wrist_fast or wrist_big
                      or leg_kick or fast_fall or lying_now)

        # ============================================================
        # YẾU TỐ HẬU QUẢ — 1 bên ngã và nằm lâu không dậy (≈70% vụ)
        # ============================================================
        aftermath = condA["lying_sustained"] or condB["lying_sustained"]
        # Tín hiệu phụ chống false positive "người ngất xỉu đứng cạnh người
        # khác": người kia bất thường / 2 bbox chồng nhau / khoảng cách dao
        # động / tăng tốc đột ngột.
        bbox_iou = self._calculate_iou(boxA, boxB)
        dist_variance = self._pair_dist_variance(objA, objB, avg_h)
        support = (
            max(condA["jitter"], condB["jitter"]) > config.CONFLICT_CALM_AGITATION_THRESH
            or bbox_iou > config.CONFLICT_GRAPPLE_IOU_THRESH
            or dist_variance > config.CONFLICT_DIST_VAR_THRESH
            or max(condA["accel"], condB["accel"]) > config.CONFLICT_ACCEL_THRESH
        )
        aftermath_ok = aftermath and (support or aggression)

        # ============================================================
        # KẾT LUẬN: (Aggression ∨ Aftermath) ∧ (Kinetic ∨ Aftermath)
        # ============================================================
        is_candidate = (aggression and kinetic) or aftermath_ok

        # Điểm confidence — kết hợp các tín hiệu theo 3 yếu tố + hậu quả
        conflict_score = (
            min(1.0, body_speed / (2 * config.CONFLICT_BODY_SPEED_THRESH)) * 1.8
            + min(1.0, closing_norm
                  / (2 * config.CONFLICT_CLOSING_NORM_THRESH)) * 0.8
            + min(1.0, wrist_attack
                  / (2 * config.CONFLICT_WRIST_ALIGNMENT_THRESH)) * 1.2
            + min(1.0, max(condA["wrist_speed"], condB["wrist_speed"]) / 40.0) * 0.6
            + min(1.0, max(condA["leg_speed"], condB["leg_speed"]) / 40.0) * 0.8
            + (condA["jitter"] + condB["jitter"]) * 0.25
            + dist_variance * 40.0 * 0.25
            + bbox_iou * 20.0 * 0.15
            + (config.CONFLICT_AFTERMATH_BONUS if aftermath_ok else 0.0)
        )

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

    def check_group_conflict(self, persons):
        """Phát hiện cụm 3 người tiếp cận/va chạm/xô xát theo temporal history.

        Trả về event khi 3 người đủ gần nhau, ít nhất một người có trạng thái
        bất thường, và mẫu chuyển động của cụm cho thấy tương tác thật.
        """
        if len(persons) < 3:
            return False, 0.0, None

        best_score = 0.0
        best_ids = None

        # Chỉ xét các cụm 3 người để giữ chi phí thấp và dễ debug.
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                for k in range(j + 1, len(persons)):
                    trio = (persons[i], persons[j], persons[k])
                    if min(len(o.bbox_history) for o in trio) < 5:
                        continue

                    boxA, boxB, boxC = [o.bbox_history[-1] for o in trio]
                    centA, centB, centC = [np.array(o.center_history[-1]) for o in trio]
                    avg_h = np.mean([
                        max(boxA[3] - boxA[1], 1e-5),
                        max(boxB[3] - boxB[1], 1e-5),
                        max(boxC[3] - boxC[1], 1e-5),
                    ])

                    # GATE: CẢ 3 phải CÙNG 1 ô lưới (vùng) trong frame.
                    # Người ở các ô khác nhau là các khu vực khác nhau → không
                    # xét như 1 cụm tương tác.
                    if len({config.grid_zone(*c) for c in (centA, centB, centC)}) != 1:
                        continue

                    pair_dists = [
                        np.linalg.norm(centA - centB) / avg_h,
                        np.linalg.norm(centA - centC) / avg_h,
                        np.linalg.norm(centB - centC) / avg_h,
                    ]

                    if max(pair_dists) > config.GROUP_INTERACTION_DIST_THRESH:
                        continue

                    conds = [self._assess_person_condition(o) for o in trio]
                    abnormal_count = sum(1 for c in conds if c["abnormal"])
                    if abnormal_count == 0:
                        continue

                    bbox_iou = max(
                        self._calculate_iou(boxA, boxB),
                        self._calculate_iou(boxA, boxC),
                        self._calculate_iou(boxB, boxC),
                    )

                    motion_strength = max(
                        max(c["wrist_speed"] for c in conds),
                        max(c["jitter"] for c in conds),
                        max(c["accel"] for c in conds),
                    )
                    pairwise_compactness = 1.0 - min(max(pair_dists), 1.0)
                    score = min(
                        0.98,
                        0.30 * abnormal_count
                        + 0.30 * pairwise_compactness
                        + 0.20 * min(motion_strength / 20.0, 1.0)
                        + 0.20 * bbox_iou,
                    )

                    if score > best_score:
                        best_score = score
                        best_ids = tuple(sorted(o.track_id for o in trio))

        if best_ids is None:
            return False, 0.0, None

        if best_ids not in self.group_conflict_state:
            self.group_conflict_state[best_ids] = 0

        if best_score > 0.0:
            self.group_conflict_state[best_ids] = min(
                config.CONFLICT_SUSTAINED_FRAMES + 2,
                self.group_conflict_state.get(best_ids, 0) + 1,
            )
        else:
            self.group_conflict_state[best_ids] = max(
                0, self.group_conflict_state.get(best_ids, 0) - 1
            )

        is_confirmed = (
            best_score > 0.0
            and self.group_conflict_state.get(best_ids, 0) >= config.CONFLICT_SUSTAINED_FRAMES
        )
        return is_confirmed, best_score, list(best_ids)

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

        lost_groups = [k for k in self.group_conflict_state
                       if any(tid not in active_track_ids for tid in k)]
        for k in lost_groups:
            del self.group_conflict_state[k]

        lost_lying = [k for k in self.lying_persist if k not in active_track_ids]
        for k in lost_lying:
            del self.lying_persist[k]

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

    def _assess_person_condition(self, obj):
        """STAGE 1: Xác định người + đánh giá TÌNH TRẠNG của người đó.

        Trả về dict chứa các tín hiệu tình trạng cá nhân (không phụ thuộc
        người kia) + cờ abnormal = người có TÌNH TRẠNG bất thường hay không.
        Bổ sung theo định nghĩa mới:
          - body_speed / body_accel : tốc độ & gia tốc CƠ THỂ (normalized).
          - wrist_velocity          : vector vận tốc cổ tay nhanh nhất (để tính
                                      hướng vung vào người kia ở cấp cặp).
          - leg_speed               : tốc độ cổ chân/ankle (cú đá/gạt chân).
          - lying_sustained         : nằm lâu không dậy (hậu quả ≈70% vụ).
        """
        wrist_speed = self._get_wrist_speed(obj)
        wrist_amplitude = self._get_wrist_amplitude(obj)
        wrist_velocity = self._get_wrist_velocity(obj)
        leg_speed = self._get_leg_speed(obj)
        jitter = self._get_bbox_jitter(obj)
        accel = self._get_sudden_accel(obj)
        lying = self._is_lying(obj)
        falling = self._is_falling(obj)
        body_speed = self._get_body_speed(obj)
        body_accel = self._get_body_accel(obj)
        lying_sustained = self._update_lying_sustain(obj, lying)

        abnormal = (
            wrist_speed > config.CONFLICT_WRIST_HIGH_THRESH
            or wrist_amplitude > config.CONFLICT_WRIST_AMPLITUDE_THRESH
            or leg_speed > config.CONFLICT_LEG_KICK_THRESH
            or jitter > config.CONFLICT_CALM_AGITATION_THRESH
            or accel > config.CONFLICT_ACCEL_THRESH
            or lying
            or falling
        )
        return {
            "wrist_speed": wrist_speed,
            "wrist_amplitude": wrist_amplitude,
            "wrist_velocity": wrist_velocity,
            "leg_speed": leg_speed,
            "body_speed": body_speed,
            "body_accel": body_accel,
            "jitter": jitter,
            "accel": accel,
            "lying": lying,
            "falling": falling,
            "lying_sustained": lying_sustained,
            "abnormal": abnormal,
        }

    def _get_body_speed(self, obj):
        """YẾU TỐ 1 — tốc độ di chuyển CƠ THỂ (normalized theo chiều cao).

        px/s chia cho chiều cao người: đánh nhau = cơ thể di chuyển nhanh
        (> đi bộ); ôm nhau / thân nhau = chậm hoặc đứng yên.
        """
        if not obj.velocity_history or not obj.bbox_history:
            return 0.0
        box = obj.bbox_history[-1]
        h = max(box[3] - box[1], 1e-5)
        return float(np.linalg.norm(obj.velocity_history[-1])) / h

    def _get_body_accel(self, obj):
        """Gia tốc CƠ THỂ normalized — lao tới / phanh gấp."""
        if not obj.accel_history or not obj.bbox_history:
            return 0.0
        box = obj.bbox_history[-1]
        h = max(box[3] - box[1], 1e-5)
        return float(np.linalg.norm(obj.accel_history[-1])) / h

    def _get_leg_speed(self, obj):
        """YẾU TỐ 2 — tốc độ cổ chân/ankle (px/s) — cú đá/gạt chân.

        Keypoints 15 (ankle trái) / 16 (ankle phải). Ôm nhau/thân nhau không
        có cử động chân nhanh.
        """
        if len(obj.pose_history) < 2:
            return 0.0
        kp_prev = obj.pose_history[-2]
        kp_curr = obj.pose_history[-1]
        if kp_prev is None or kp_curr is None:
            return 0.0
        dt = max(obj.time_history[-1] - obj.time_history[-2], 1e-5)
        speeds = []
        for idx in (15, 16):
            if (idx < len(kp_curr) and idx < len(kp_prev)
                    and kp_curr[idx][2] > 0.3 and kp_prev[idx][2] > 0.3):
                cur = np.asarray(kp_curr[idx][:2], dtype=float)
                prev = np.asarray(kp_prev[idx][:2], dtype=float)
                speeds.append(float(np.linalg.norm(cur - prev)) / dt)
        return max(speeds, default=0.0)

    def _get_wrist_velocity(self, obj):
        """Vector vận tốc cổ tay di chuyển NHANH NHẤT (px/s).

        Trả None nếu chưa đủ pose. Dùng ở cấp cặp để chiếu lên hướng tới
        người kia (vung tay đấm/đẩy) — khác với ôm/vỗ (cổ tay không hướng
        tới người kia).
        """
        if len(obj.pose_history) < 2:
            return None
        kp_prev = obj.pose_history[-2]
        kp_curr = obj.pose_history[-1]
        if kp_prev is None or kp_curr is None:
            return None
        dt = max(obj.time_history[-1] - obj.time_history[-2], 1e-5)
        best = None
        best_mag = 0.0
        for idx in (9, 10):  # Wrist trái (9) / phải (10)
            if (idx < len(kp_curr) and idx < len(kp_prev)
                    and kp_curr[idx][2] > 0.3 and kp_prev[idx][2] > 0.3):
                cur = np.asarray(kp_curr[idx][:2], dtype=float)
                prev = np.asarray(kp_prev[idx][:2], dtype=float)
                v = (cur - prev) / dt
                m = float(np.linalg.norm(v))
                if m > best_mag:
                    best_mag = m
                    best = v
        return best

    def _update_lying_sustain(self, obj, lying):
        """HẬU QUẢ — đếm số frame liên tiếp ở tư thế nằm (nằm lâu = không dậy).

        Chỉ được coi là "nằm lâu không dậy" khi duy trì >= CONFLICT_LYING_SUSTAINED_FRAMES.
        """
        tid = obj.track_id
        if lying:
            self.lying_persist[tid] = min(
                config.CONFLICT_LYING_SUSTAINED_FRAMES + 2,
                self.lying_persist.get(tid, 0) + 1,
            )
        else:
            self.lying_persist[tid] = max(0, self.lying_persist.get(tid, 0) - 1)
        return self.lying_persist[tid] >= config.CONFLICT_LYING_SUSTAINED_FRAMES

    def _get_wrist_attack_component(self, condA, condB, centA, centB):
        """YẾU TỐ 2 — thành phần vận tốc cổ tay HƯỚNG VÀO người kia (px/s).

        Chiếu vector vận tốc cổ tay lên đường nối từ người này tới người kia.
        Giá trị dương lớn = cổ tay đang lao VỀ phía người kia (đấm/đẩy). Ôm
        nhau/vỗ lưng: cổ tay đi ngang hoặc không hướng tới người kia → thấp.
        """
        def component(wrist_vel, own_center, opp_center):
            if wrist_vel is None:
                return 0.0
            d = opp_center - own_center
            n = float(np.linalg.norm(d))
            if n < 1e-5:
                return 0.0
            return float(np.dot(wrist_vel, d / n))

        return max(
            component(condA["wrist_velocity"], centA, centB),
            component(condB["wrist_velocity"], centB, centA),
        )

    def _pair_dist_variance(self, objA, objB, avg_h):
        """Phương sai khoảng cách 2 người (theo avg height) — giằng co.

        Trong đánh nhau, 2 người lúc áp sát lúc tách xa → phương sai lớn.
        Nói chuyện/đứng yên/đi chung → khoảng cách ổn định.
        """
        window = min(8, min(len(objA.center_history), len(objB.center_history)))
        dists = [
            np.linalg.norm(np.array(cA) - np.array(cB)) / avg_h
            for cA, cB in zip(
                list(objA.center_history)[-window:],
                list(objB.center_history)[-window:],
            )
        ]
        return float(np.var(dists)) if len(dists) > 1 else 0.0

    def _get_closing_speed(self, objA, objB):
        """YẾU TỐ 3 — tốc độ 2 người lao vào nhau (px/s).

        Chiếu vận tốc tương đối lên đường nối 2 tâm. Dương = đang tiến lại gần.
        Ôm/thân nhau tiếp cận chậm → thấp.
        """
        if not objA.velocity_history or not objB.velocity_history:
            return 0.0
        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        d = centB - centA
        dist = float(np.linalg.norm(d))
        if dist < 1e-5:
            return 0.0
        vA = np.array(objA.velocity_history[-1])
        vB = np.array(objB.velocity_history[-1])
        return max(0.0, float(np.dot(vA - vB, d / dist)))

    def _get_wrist_amplitude(self, obj):
        """BIÊN ĐỘ vung tay (px) — cổ tay vung XA bao nhiêu trong cửa sổ.

        Khác _get_wrist_speed (đo tốc độ tức thời), amplitude đo QUÃNG đường
        cổ tay di chuyển xa nhất trong cửa sổ pose. Người vung tay đấm/vẫy
        mạnh có biên độ lớn dù tốc độ 2 frame kề nhau có thể thấp.
        """
        if len(obj.pose_history) < 3:
            return 0.0
        window = min(6, len(obj.pose_history))
        poses = [kp for kp in list(obj.pose_history)[-window:] if kp is not None]
        if len(poses) < 2:
            return 0.0

        max_amp = 0.0
        for idx in (9, 10):  # Wrist trái (9) / phải (10)
            pts = []
            for kp in poses:
                if idx < len(kp) and kp[idx][2] > 0.3:
                    pts.append(np.asarray(kp[idx][:2], dtype=float))
            if len(pts) >= 2:
                arr = np.asarray(pts)
                # Biên độ = khoảng cách xa nhất giữa 2 vị trí cổ tay trong cửa sổ
                span = float(np.max(np.linalg.norm(
                    arr[:, None, :] - arr[None, :, :], axis=2)))
                max_amp = max(max_amp, span)
        return max_amp

    def _get_sudden_accel(self, obj):
        """TĂNG TỐC/giảm tốc đột ngột (px/s²) — lao tới / giật người.

        Dùng độ lớn gia tốc lớn nhất trong cửa sổ ngắn; người đi bộ/đứng yên
        có gia tốc nhỏ, lao tới/phanh gấp tạo đỉnh gia tốc rõ rệt.
        """
        if len(obj.accel_history) == 0:
            return 0.0
        window = min(4, len(obj.accel_history))
        accels = [float(np.linalg.norm(a)) for a in list(obj.accel_history)[-window:]]
        return max(accels)

    def _is_lying(self, obj):
        """Tư thế NẰM VẬT — aspect ratio ngang (w/h lớn) như người bị quật
        ngã/đè xuống đất khi xô xát."""
        if len(obj.bbox_history) == 0:
            return False
        box = obj.bbox_history[-1]
        w = box[2] - box[0]
        h = max(box[3] - box[1], 1e-5)
        return (w / h) > config.CONFLICT_LYING_ASPECT_THRESH

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
