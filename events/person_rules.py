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
        """Phát hiện 2 người xô xát/giằng co/đánh nhau — Pipeline 2 GIAI ĐOẠN.

        STAGE 1: XÁC ĐỊNH NGƯỜI + đánh giá TÌNH TRẠNG của từng người:
          - Biên độ vung tay (wrist amplitude): cổ tay vung XA bao nhiêu.
          - Tốc độ cổ tay (wrist speed): vung NHANH như thế nào.
          - Tăng tốc đột ngột (sudden accel): lao tới / giật người.
          - Tư thế nằm vật nhau (lying): 1 người bị quật ngã/đè xuống.
          - Độ giật cục (jitter): cử động giật cục mạnh.
        STAGE 2: KẾT LUẬN XÔ XÁT CHỈ KHI có TÌNH TRẠNG BẤT THƯỜNG + gần nhau.

        LOẠI BỎ "ở sát nhau": 2 người đứng nói chuyện / ôm nhau / đứng chờ,
        gần nhau NHƯNG cả hai đều bình thường (không vung tay, không giật,
        không tăng tốc, không nằm) → KHÔNG phải xô xát dù khoảng cách rất gần.

        ĐƯỜNG A (Pose): 1 người vung cổ tay nhanh hoặc biên độ lớn + gần nhau +
        (người kia cũng động HOẶC khoảng cách dao động).
        ĐƯỜNG B (BBox fallback): CẢ HAI đều cử động giật cục mạnh.
        ĐƯỜNG C (Grapple/nằm vật): 2 người chồng bbox + (giằng co HOẶC tư thế nằm).
        ĐƯỜNG D (Một chiều): 1 người kích động cực mạnh hoặc tăng tốc đột ngột
        khi rất gần người kia.
        ĐƯỜNG E (Người ngã trong cặp): 1 người ngã + KẾT HỢP thêm bằng chứng
        xô xát (người kia bất thường, bbox chồng nhau, khoảng cách dao động,
        hoặc tăng tốc đột ngột). Chỉ "đứng gần nhau + có người ngã" thì KHÔNG
        tính là đánh nhau.
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
        # vùng). Khác ô = 2 khu vực xa nhau trong khung hình → không xét
        # xô xát dù khoảng cách tương đối (theo chiều cao) vẫn dưới ngưỡng.
        # ============================================================
        if config.grid_zone(*centA) != config.grid_zone(*centB):
            self.conflict_state[pair_key] = max(
                0, self.conflict_state.get(pair_key, 0) - 1
            )
            return False, 0.0

        # ============================================================
        # STAGE 1: XÁC ĐỊNH NGƯỜI + TÌNH TRẠNG của TỪNG người
        # ============================================================
        condA = self._assess_person_condition(objA)
        condB = self._assess_person_condition(objB)

        # ============================================================
        # LOẠI BỎ "ở sát nhau": CẢ HAI đều bình thường -> không xô xát
        # ============================================================
        if not (condA["abnormal"] or condB["abnormal"]):
            self.conflict_state[pair_key] = max(
                0, self.conflict_state.get(pair_key, 0) - 1
            )
            return False, 0.0

        # Độ kích động của mỗi người = max(wrist speed, bbox jitter)
        agitation_A = max(condA["wrist_speed"], condA["jitter"])
        agitation_B = max(condB["wrist_speed"], condB["jitter"])
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
        # STAGE 2: QUYẾT ĐỊNH — kết hợp tình trạng cá nhân (Stage 1)
        # ============================================================
        # Đường A — Pose: vung cổ tay nhanh (speed) HOẶC biên độ lớn (amp)
        pose_punch = (
            max(condA["wrist_speed"], condB["wrist_speed"])
            > config.CONFLICT_WRIST_HIGH_THRESH
            or max(condA["wrist_amplitude"], condB["wrist_amplitude"])
            > config.CONFLICT_WRIST_AMPLITUDE_THRESH
        )

        # Đường B — BBox: CẢ HAI người cử động giật cục rất mạnh.
        both_jittery = (min(condA["jitter"], condB["jitter"])
                        > config.CONFLICT_BBOX_JITTER_THRESH)

        # Đường C — Grapple (vật lộn tại chỗ / nằm vật nhau): 2 người dính sát,
        # bbox chồng nhau (IoU cao) + (1 người giằng co đáng kể HOẶC có người
        # ở tư thế nằm ngang — bị quật ngã/đè xuống).
        grapple_overlap = (bbox_iou > config.CONFLICT_GRAPPLE_IOU_THRESH
                           and dist < config.CONFLICT_GRAPPLE_DIST_THRESH)
        grapple_activity = (max(condA["jitter"], condB["jitter"])
                            > config.CONFLICT_GRAPPLE_MIN_JITTER
                            or condA["lying"] or condB["lying"])

        # Tương tác: người kia cũng động hoặc khoảng cách dao động mạnh
        signal_mutual = mutual_agitation > config.CONFLICT_MUTUAL_AGITATION_THRESH
        signal_oscillation = dist_variance > config.CONFLICT_DIST_VAR_THRESH

        # Đường A: vung tay + có tương tác
        path_pose = pose_punch and (signal_mutual or signal_oscillation)

        # Đường B: cả 2 giật mạnh + có tương tác (đôi công / giằng co sát nhau)
        path_bbox = both_jittery and (signal_mutual or signal_oscillation)

        # Đường C: vật lộn/nằm vật — 2 người CHỒNG bbox + có người giằng co
        # hoặc ở tư thế nằm. Ôm nhau: grapple_activity fail (jitter≈0, không
        # nằm). Đi ngang: grapple_overlap fail (IoU thấp).
        path_grapple = (grapple_overlap and grapple_activity)

        # Đường D: MỘT CHIỀU tấn công — 1 người cực kỳ kích động (vung tay/
        # giật) HOẶC tăng tốc đột ngột khi rất gần người kia (đấm/đẩy dồn dập).
        top_agitation = max(agitation_A, agitation_B)
        path_one_sided = (
            (top_agitation > config.CONFLICT_ONE_SIDED_AGITATION_THRESH
             or max(condA["accel"], condB["accel"])
             > config.CONFLICT_ACCEL_THRESH)
            and dist < config.CONFLICT_ONE_SIDED_DIST_THRESH
            and (signal_oscillation or grapple_overlap)
        )

        # Đường E — NGƯỜI NGÃ trong cặp: ít nhất 1 người có tín hiệu ngã.
        # KHÔNG đủ nếu chỉ "gần nhau + có người ngã" (người đứng gần có thể
        # đang ngất xỉu/ngã bệnh, không phải đánh nhau). Phải KẾT HỢP thêm
        # ít nhất 1 điều kiện xô xát:
        #   (1) Người kia cũng BẤT THƯỜNG (kích động: vung tay/giật/tăng tốc
        #       hoặc cũng ngã) — đôi công / đánh trả.
        #   (2) 2 người CHỒNG BBOX (IoU cao) — vật lộn/đè/đánh gục.
        #   (3) Khoảng cách DAO ĐỘNG mạnh — giằng co trước khi ngã.
        #   (4) Tăng tốc/giảm tốc ĐỘT NGỘT — cú đấm/cú đẩy khiến người kia ngã.
        fall_in_pair = condA["falling"] or condB["falling"]
        attacker_abnormal = (condB["abnormal"] if condA["falling"]
                             else condA["abnormal"])
        path_fall = fall_in_pair and (
            attacker_abnormal
            or bbox_iou > config.CONFLICT_GRAPPLE_IOU_THRESH
            or signal_oscillation
            or max(condA["accel"], condB["accel"]) > config.CONFLICT_ACCEL_THRESH
        )

        is_candidate = (path_pose or path_bbox or path_grapple or path_one_sided
                        or path_fall)

        # Điểm kết hợp (chỉ để tính confidence)
        # Tăng trọng số để xô xát CONFIRMED vượt MIN_ALERT_CONFIDENCE (0.9).
        # Trước đây vật lộn tại chỗ chỉ đạt ~0.83 → bị lọc sạch ở main.py.
        # Path E cộng bonus cố định để người ngã trong cặp luôn vượt ngưỡng.
        conflict_score = (max(condA["wrist_speed"], condB["wrist_speed"]) * 0.30
                          + max(condA["wrist_amplitude"],
                                condB["wrist_amplitude"]) * 0.10
                          + (condA["jitter"] + condB["jitter"]) * 0.30
                          + max(condA["accel"], condB["accel"]) * 0.05
                          + dist_variance * 40.0 * 0.25
                          + mutual_agitation * 0.15
                          + bbox_iou * 20.0 * 0.15
                          + (1.0 if path_fall else 0.0)
                          * config.CONFLICT_FALL_SCORE_BONUS)

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
        """
        wrist_speed = self._get_wrist_speed(obj)
        wrist_amplitude = self._get_wrist_amplitude(obj)
        jitter = self._get_bbox_jitter(obj)
        accel = self._get_sudden_accel(obj)
        lying = self._is_lying(obj)
        falling = self._is_falling(obj)

        abnormal = (
            wrist_speed > config.CONFLICT_WRIST_HIGH_THRESH
            or wrist_amplitude > config.CONFLICT_WRIST_AMPLITUDE_THRESH
            or jitter > config.CONFLICT_CALM_AGITATION_THRESH
            or accel > config.CONFLICT_ACCEL_THRESH
            or lying
            or falling
        )
        return {
            "wrist_speed": wrist_speed,
            "wrist_amplitude": wrist_amplitude,
            "jitter": jitter,
            "accel": accel,
            "lying": lying,
            "falling": falling,
            "abnormal": abnormal,
        }

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
