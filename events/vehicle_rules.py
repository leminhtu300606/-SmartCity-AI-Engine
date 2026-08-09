import numpy as np
import config


class VehicleAccidentRules:
    """Phát hiện va chạm xe bằng Accident Classifier (multi-feature).

    Pipeline:
      Camera → Object Detection → Vehicle Tracking → Trajectory/Speed/Distance
      → Collision Feature Extraction → Accident Classifier → Có/Không va chạm

    Các đặc trưng:
      1. Proximity  : 2 xe tiến rất gần / bounding box giao nhau (gate bắt buộc).
      2. Closing    : khoảng cách giảm nhanh (closing velocity).
      3. Dist drop  : khoảng cách tụt nhanh frame-to-frame.
      4. Direction  : hướng/chuyển động thay đổi đột ngột.
      5. Decel      : giảm tốc đột ngột.
      6. Post-contact: 1 xe dừng hoặc đổi hướng bất thường NGAY SAU tiếp xúc.

    KHÔNG chỉ dựa vào bbox overlap: xe chạy sát nhau / bị che khuất cũng tạo
    overlap giả → proximity là điều kiện cần nhưng CHƯA đủ; phải kèm các đặc
    trưng động học và được xác nhận qua nhiều frame (sustained).
    """

    def __init__(self):
        # Temporal state cho sustained confirmation
        self.collision_state = {}     # (min_id, max_id) -> sustained_count
        self.hard_stop_state = {}     # track_id -> sustained_count
        self.pair_dist_hist = {}      # (min_id, max_id) -> deque(distances)
        self.pair_contact_frame = {}  # (min_id, max_id) -> frame gần nhất đạt proximity

        # Vehicle ↔ Object / Falling object
        self.object_collision_state = {}   # (min_id, max_id) -> sustained_count
        self.object_fall_state = {}        # (min_id, max_id) -> sustained_count
        self.object_fall_vy_hist = {}      # track_id -> [vy_norm, ...] quỹ đạo rơi

    def check_collision(self, objA, objB):
        """Accident Classifier: trích xuất đặc trưng -> chấm điểm -> quyết định."""
        if len(objA.bbox_history) < 3 or len(objB.bbox_history) < 3:
            return False, 0.0

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        pair_key = (min(objA.track_id, objB.track_id),
                    max(objA.track_id, objB.track_id))

        # ------------------------------------------------------------
        # FEATURE 1: Proximity — 2 xe tiến rất gần / bbox giao nhau
        # Chuẩn hóa theo đường chéo trung bình (không dùng pixel tuyệt đối).
        # ------------------------------------------------------------
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
        prox_score = min(1.0, iou * 2.0 + max(0.0, 1.0 - rel_dist
                                              / config.VEHICLE_PROXIMITY_DIST_RATIO))

        # ------------------------------------------------------------
        # FEATURE 2: Closing velocity — khoảng cách giảm nhanh
        # ------------------------------------------------------------
        closing_speed = self._get_closing_speed(objA, objB)
        closing_score = min(1.0, closing_speed / config.VEHICLE_CLOSING_SPEED_THRESH)

        # ------------------------------------------------------------
        # FEATURE 3: Dist drop — khoảng cách tụt nhanh frame-to-frame
        # ------------------------------------------------------------
        dist_drop = self._get_dist_drop(pair_key, dist)
        dist_drop_score = min(1.0, dist_drop / config.VEHICLE_DIST_DROP_THRESH)

        # ------------------------------------------------------------
        # FEATURE 4: Direction change — đổi hướng / chuyển động đột ngột
        # ------------------------------------------------------------
        dir_changeA = self._get_direction_change(objA)
        dir_changeB = self._get_direction_change(objB)
        direction_score = max(
            min(1.0, dir_changeA / config.VEHICLE_DIR_CHANGE_THRESH),
            min(1.0, dir_changeB / config.VEHICLE_DIR_CHANGE_THRESH),
        )

        # ------------------------------------------------------------
        # FEATURE 5: Deceleration — giảm tốc đột ngột
        # ------------------------------------------------------------
        decelA = self._get_deceleration(objA)
        decelB = self._get_deceleration(objB)
        decel_score = max(
            min(1.0, decelA / config.VEHICLE_DECEL_THRESH),
            min(1.0, decelB / config.VEHICLE_DECEL_THRESH),
        )

        # "Dừng đột ngột": vận tốc tức thời nhảy về ~0 trong 1 frame (va chạm
        # mạnh) làm _get_deceleration trả 0 (vel≈0 bị bỏ qua). Bắt tín hiệu này
        # từ lịch sử vận tốc: trước đó chạy nhanh, bây giờ đứng yên.
        sharp_stop = max(self._get_sharp_stop(objA), self._get_sharp_stop(objB))
        impact_score = max(decel_score, sharp_stop)

        # ------------------------------------------------------------
        # FEATURE 6: Post-contact — dừng/đổi hướng bất thường sau TIẾP XÚC THẬT
        # ------------------------------------------------------------
        # CHỈ arm cửa sổ hậu va chạm khi bbox 2 xe THỰC SỰ chồng lên nhau
        # (overlap), KHÔNG phải chỉ "gần nhau". Xe chạy sát / vượt gần nhau không
        # overlap → không bao giờ kích nhánh post-contact → bỏ false positive
        # do xe quẹo / dừng đèn đỏ cạnh xe khác.
        has_overlap = iou > config.VEHICLE_IOU_THRESH
        self._update_contact_frame(pair_key, has_overlap)
        post_contact_score = self._get_post_contact_anomaly(
            pair_key, objA, objB, dir_changeA, dir_changeB
        )

        # ------------------------------------------------------------
        # ACCIDENT CLASSIFIER: điểm tổng hợp có trọng số
        # ------------------------------------------------------------
        w = config.VEHICLE_COLLISION_WEIGHTS
        score = (
            w["proximity"] * prox_score
            + w["closing"] * closing_score
            + w["dist_drop"] * dist_drop_score
            + w["direction"] * direction_score
            + w["decel"] * impact_score
            + w["post_contact"] * post_contact_score
        )

        # Gate bắt buộc: phải gần nhau (proximity là điều kiện cần) + phải có
        # TÍN HIỆU ĐỘNG HỌC thật. Xe đậu sát nhau / đi song song (closing≈0,
        # decel≈0) → KHÔNG phải va chạm dù bbox rất gần.
        kinetic_signal = max(closing_score, impact_score)
        has_kinetic = kinetic_signal > config.VEHICLE_COLLISION_MIN_KINETIC

        is_proximate_candidate = (is_proximate and has_kinetic
                                  and score > config.VEHICLE_COLLISION_SCORE_THRESH)

        # Cho phép "hậu va chạm": ngay sau thời điểm TIẾP XÚC THẬT (bbox overlap),
        # xe bị hất lệch (đổi hướng đột ngột) hoặc dừng bất thường dù 2 xe đã bắt
        # đầu tách xa. Cửa sổ chỉ arm khi có overlap thật (xem _update_contact_frame).
        post_contact_only = (
            pair_key in self.pair_contact_frame
            and post_contact_score > 0.7
            and score > config.VEHICLE_COLLISION_SCORE_THRESH * 0.8
        )

        is_candidate = is_proximate_candidate or post_contact_only

        # Temporal sustained confirmation (tránh spike 1 frame)
        # Cap counter tại SUSTAINED + 2 để cảnh báo tắt nhanh khi hết hiện tượng.
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
        return is_confirmed, score

    def check_object_collision(self, objV, objO):
        """Xe va chạm vật thể / người / xe 2 bánh (VEHICLE_OBJECT_COLLISION).

        Khác check_collision (xe-xe): chỉ cần MỘT bên là xe (car/bus/truck),
        bên kia là bất kỳ object nào — kể cả người (class 0), motorbike...
        Vật thể nhỏ hơn xe nên:
          - Proximity dùng dist/avg_diagonal (avg_diag thấp hơn -> tỉ lệ nhạy).
          - Closing speed ngưỡng thấp hơn (vật thể nhẹ, va nhanh).
          - IoU threshold thấp hơn (bbox vật thể nhỏ so với xe).
        """
        if len(objV.bbox_history) < 3 or len(objO.bbox_history) < 3:
            return False, 0.0

        boxV, boxO = objV.bbox_history[-1], objO.bbox_history[-1]
        pair_key = (min(objV.track_id, objO.track_id),
                    max(objV.track_id, objO.track_id))

        # ------------------------------------------------------------
        # PROXIMITY — gate bắt buộc
        # ------------------------------------------------------------
        diagV = np.sqrt((boxV[2] - boxV[0]) ** 2 + (boxV[3] - boxV[1]) ** 2)
        diagO = np.sqrt((boxO[2] - boxO[0]) ** 2 + (boxO[3] - boxO[1]) ** 2)
        avg_diag = max((diagV + diagO) / 2.0, 1e-5)

        centV = np.array(objV.center_history[-1])
        centO = np.array(objO.center_history[-1])
        dist = float(np.linalg.norm(centV - centO))
        rel_dist = dist / avg_diag
        iou = self._calculate_iou(boxV, boxO)

        is_proximate = (iou > config.VEHICLE_OBJECT_IOU_THRESH
                        or rel_dist < config.VEHICLE_OBJECT_PROXIMITY_DIST_RATIO)
        prox_score = min(1.0, iou * 2.0 + max(0.0, 1.0 - rel_dist
                                              / config.VEHICLE_OBJECT_PROXIMITY_DIST_RATIO))

        # ------------------------------------------------------------
        # KINETIC SIGNALS — động học của va chạm
        # ------------------------------------------------------------
        closing = self._get_closing_speed(objV, objO)
        closing_score = min(1.0, closing / config.VEHICLE_OBJECT_CLOSING_SPEED_THRESH)

        # Vật thể nhỏ có thể bị hất lệch mạnh khi va chạm (đổi hướng đột ngột)
        dirV = self._get_direction_change(objV)
        dirO = self._get_direction_change(objO)
        direction_score = max(
            min(1.0, dirV / config.VEHICLE_DIR_CHANGE_THRESH),
            min(1.0, dirO / config.VEHICLE_DIR_CHANGE_THRESH),
        )

        decelV = self._get_deceleration(objV)
        decelO = self._get_deceleration(objO)
        decel_score = max(
            min(1.0, decelV / config.VEHICLE_DECEL_THRESH),
            min(1.0, decelO / config.VEHICLE_DECEL_THRESH),
        )
        # Xe dừng đột ngột (va chạm mạnh) — bắt cả khi vận tốc nhảy về 0 trong 1 frame
        sharp_stop = self._get_sharp_stop(objV)

        # Người đang chạy nhanh / chuyển động bất thường gần xe (băng qua đường)
        speedO = float(np.linalg.norm(np.array(objO.velocity_history[-1])))
        speedO_score = min(1.0, speedO / config.PERSON_ABNORMAL_SPEED_THRESH)

        # ------------------------------------------------------------
        # ACCIDENT CLASSIFIER — điểm tổng hợp
        # ------------------------------------------------------------
        score = (
            prox_score * 0.35
            + closing_score * 0.25
            + direction_score * 0.15
            + max(decel_score, sharp_stop) * 0.15
            + speedO_score * 0.10
        )

        # Gate: phải gần nhau + có tín hiệu động học thật (siết chặt hơn 0.15)
        kinetic_signal = max(closing_score, decel_score, speedO_score, sharp_stop)
        has_kinetic = kinetic_signal > 0.25

        is_candidate = (is_proximate and has_kinetic
                        and score > config.VEHICLE_OBJECT_SCORE_THRESH)

        if is_candidate:
            self.object_collision_state[pair_key] = min(
                config.VEHICLE_OBJECT_SUSTAINED + 2,
                self.object_collision_state.get(pair_key, 0) + 1,
            )
        else:
            self.object_collision_state[pair_key] = max(
                0, self.object_collision_state.get(pair_key, 0) - 1
            )

        is_confirmed = (is_candidate
                        and self.object_collision_state.get(pair_key, 0)
                        >= config.VEHICLE_OBJECT_SUSTAINED)
        return is_confirmed, score

    def check_object_falling(self, objV, objO):
        """Vật thể rơi từ trên xuống trúng xe (OBJECT_FALLING_ON_VEHICLE).

        Tín hiệu:
          1. Quỹ đạo rơi: vận tốc dọc (vy) hướng xuống, |vy|/chiều cao object
             vượt ngưỡng (rơi nhanh, không phải đi bộ/leo).
          2. Chồng lấn: bbox vật thể rơi PHỦ LÊN bbox xe (overlap ratio cao) —
             vật đang ở vị trí trúng xe.
          Cả 2 duy trì >= OBJECT_FALL_SUSTAINED frames.

        Phân biệt với collision: collision = vật đến TỪ MẶT ĐẤT tiến lại gần xe
        (chuyển động ngang). Falling = vật rơi TỪ TRÊN XUỐNG (chuyển động dọc
        chiếm ưu thế, |vy| >> |vx|).
        """
        if len(objV.bbox_history) < 4 or len(objO.bbox_history) < 4:
            return False, 0.0

        boxV, boxO = objV.bbox_history[-1], objO.bbox_history[-1]
        pair_key = (min(objV.track_id, objO.track_id),
                    max(objV.track_id, objO.track_id))

        # ------------------------------------------------------------
        # 1) Quỹ đạo rơi — dùng trung bình vy trong cửa sổ
        # ------------------------------------------------------------
        vy_norm = self._get_falling_velocity(objO)
        if vy_norm is None:
            return False, 0.0

        is_falling = (vy_norm > config.OBJECT_FALL_VY_NORM_THRESH)

        # ------------------------------------------------------------
        # 2) Overlap với bbox xe — vật thể đang ở vị trí trúng xe
        # ------------------------------------------------------------
        overlap_ratio = self._fall_overlap_ratio(boxV, boxO)
        is_overlapping = (overlap_ratio > config.OBJECT_FALL_OVERLAP_RATIO)

        is_candidate = is_falling and is_overlapping

        if is_candidate:
            self.object_fall_state[pair_key] = min(
                config.OBJECT_FALL_SUSTAINED + 2,
                self.object_fall_state.get(pair_key, 0) + 1,
            )
        else:
            self.object_fall_state[pair_key] = max(
                0, self.object_fall_state.get(pair_key, 0) - 1
            )

        is_confirmed = (is_candidate
                        and self.object_fall_state.get(pair_key, 0)
                        >= config.OBJECT_FALL_SUSTAINED)
        # Confidence: rơi nhanh + chồng lấn sâu
        score = min(0.98, 0.6 + min(vy_norm / 4.0, 0.25) + overlap_ratio * 0.15)
        return is_confirmed, score

    def check_hard_stop(self, obj):
        """Phát hiện xe dừng bất thường với temporal confirmation.

        Kiểm tra: đang chạy nhanh → dừng đột ngột (có giai đoạn giảm tốc mạnh).
        """
        history_len = config.VEHICLE_HARD_STOP_HISTORY
        if len(obj.velocity_history) < history_len:
            return False

        speeds = [np.linalg.norm(v) for v in list(obj.velocity_history)[-history_len:]]

        # Tốc độ trong nửa đầu cửa sổ phải cao, nửa cuối phải thấp
        half = history_len // 2
        max_prior_speed = max(speeds[:half])
        current_speed = min(speeds[-3:])  # Vài frame cuối phải đều chậm

        is_stopped = (current_speed < config.VEHICLE_HARD_STOP_SPEED_LOW
                      and max_prior_speed > config.VEHICLE_HARD_STOP_SPEED_HIGH)

        # Kiểm tra có giai đoạn giảm tốc mạnh trong cửa sổ
        has_sharp_decel = False
        for i in range(1, len(speeds)):
            if speeds[i - 1] > 15.0 and speeds[i] < 5.0:
                has_sharp_decel = True
                break

        is_candidate = is_stopped and has_sharp_decel

        # Sustained confirmation
        # Cap counter tại SUSTAINED + 2 để cảnh báo tắt nhanh khi xe chạy lại.
        tid = obj.track_id
        if is_candidate:
            self.hard_stop_state[tid] = min(
                config.VEHICLE_HARD_STOP_SUSTAINED + 2,
                self.hard_stop_state.get(tid, 0) + 1,
            )
        else:
            self.hard_stop_state[tid] = max(0, self.hard_stop_state.get(tid, 0) - 1)

        return (is_candidate
                and self.hard_stop_state.get(tid, 0)
                >= config.VEHICLE_HARD_STOP_SUSTAINED)

    # ----------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------

    def _get_dist_drop(self, pair_key, current_dist):
        """Độ tụt khoảng cách frame-to-frame (px) — bắt 'tiến lại nhanh'."""
        hist = self.pair_dist_hist.setdefault(pair_key, [])
        if len(hist) == 0:
            hist.append(current_dist)
            return 0.0
        prev = hist[-1]
        hist.append(current_dist)
        if len(hist) > 8:
            hist.pop(0)
        return max(0.0, prev - current_dist)

    def _update_contact_frame(self, pair_key, has_overlap):
        """Ghi nhận frame gần nhất mà 2 xe THỰC SỰ chồng bbox (tiếp xúc thật)."""
        if has_overlap:
            self.pair_contact_frame[pair_key] = config.VEHICLE_POST_CONTACT_WINDOW
        else:
            if pair_key in self.pair_contact_frame:
                self.pair_contact_frame[pair_key] -= 1
                if self.pair_contact_frame[pair_key] <= 0:
                    del self.pair_contact_frame[pair_key]

    def _get_post_contact_anomaly(self, pair_key, objA, objB,
                                   dir_changeA, dir_changeB):
        """Đặc trưng hậu va chạm: 1 xe dừng / đổi hướng NGAY SAU tiếp xúc thật.

        Chỉ chạy khi cửa sổ TIẾP XÚC THẬT (bbox overlap) đang mở. Nếu có dấu
        hiệu: một xe đổi hướng đột ngột, hoặc tốc độ hiện tại rất thấp (dừng lại)
        trong khi trước đó đang chạy nhanh → nghiêng về tai nạn thật.
        """
        if pair_key not in self.pair_contact_frame:
            return 0.0

        # Đổi hướng đột ngột sau tiếp xúc
        strong_turn = max(dir_changeA, dir_changeB) > config.VEHICLE_DIR_CHANGE_THRESH

        # Một xe dừng hẳn sau khi đã chạy nhanh
        stopped_after_fast = 0.0
        for obj in (objA, objB):
            if len(obj.velocity_history) < 4:
                continue
            speeds = [np.linalg.norm(v) for v in list(obj.velocity_history)[-4:]]
            now_speed = np.linalg.norm(obj.velocity_history[-1])
            prev_fast = max(speeds[:-1]) > 15.0
            if prev_fast and now_speed < config.VEHICLE_HARD_STOP_SPEED_LOW:
                stopped_after_fast = 1.0

        if strong_turn:
            return 0.9
        return stopped_after_fast

    def _get_falling_velocity(self, obj):
        """Vận tốc dọc chuẩn hóa (|vy|/chiều cao object) — phát hiện quỹ đạo rơi.

        Chỉ xét khi chuyển động dọc CHIẾM ƯU THẾ so với ngang (vật rơi không phải
        chạy ngang). Dùng trung bình trong cửa sổ để chống nhiễu tracking.
        Trả về None nếu chưa đủ dữ liệu.
        """
        if len(obj.velocity_history) < 2 or len(obj.bbox_history) == 0:
            return None

        window = min(config.OBJECT_FALL_WINDOW, len(obj.velocity_history))
        vys = [v[1] for v in list(obj.velocity_history)[-window:]]
        vxs = [v[0] for v in list(obj.velocity_history)[-window:]]

        avg_vy = float(np.mean(vys))
        avg_vx = float(np.mean(vxs))
        # Rơi xuống: vy dương (y tăng về phía dưới). |vy| phải > 1.5*|vx|
        if avg_vy <= 0 or abs(avg_vx) > abs(avg_vy) * 1.5:
            return 0.0

        box = obj.bbox_history[-1]
        h = max(box[3] - box[1], 1e-5)
        return avg_vy / h

    def _fall_overlap_ratio(self, boxV, boxO):
        """Tỷ lệ diện tích bbox vật thể rơi nằm TRONG bbox xe.

        Vật thể rơi trúng xe → phần lớn bbox của nó nằm trong vùng xe.
        """
        xA = max(boxV[0], boxO[0])
        yA = max(boxV[1], boxO[1])
        xB = min(boxV[2], boxO[2])
        yB = min(boxV[3], boxO[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaO = (boxO[2] - boxO[0]) * (boxO[3] - boxO[1])
        return inter / max(areaO, 1e-5)

    def _get_closing_speed(self, objA, objB):
        """Tốc độ tiến lại gần nhau (chiếu relative velocity lên đường nối 2 tâm)."""
        if len(objA.velocity_history) == 0 or len(objB.velocity_history) == 0:
            return 0.0

        centA = np.array(objA.center_history[-1])
        centB = np.array(objB.center_history[-1])
        direction = centB - centA
        dist = np.linalg.norm(direction)

        velA = np.array(objA.velocity_history[-1])
        velB = np.array(objB.velocity_history[-1])
        rel_vel = velA - velB

        if dist < 1e-5:
            # Hai tâm trùng nhau (thời điểm va chạm): dùng hướng vận tốc tương đối
            # làm fallback, không để closing speed tụt về 0 giữa lúc tiếp xúc.
            rv_norm = np.linalg.norm(rel_vel)
            if rv_norm < 1e-5:
                return 0.0
            direction = rel_vel / rv_norm
        else:
            direction = direction / dist

        # Closing speed = relative velocity projected onto connecting line
        # Dương = đang tiến lại gần nhau
        closing = float(np.dot(rel_vel, direction))
        return max(0.0, closing)

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def _get_deceleration(self, obj):
        if len(obj.accel_history) < 2:
            return 0.0
        acc = obj.accel_history[-1]
        vel = obj.velocity_history[-1]
        # Bỏ qua nếu xe đang đứng yên
        if np.linalg.norm(vel) < 1e-3:
            return 0.0
        # Chiếu gia tốc ngược hướng vận tốc = Giảm tốc
        decel = -np.dot(acc, vel) / np.linalg.norm(vel)
        return max(0.0, decel)

    def _get_sharp_stop(self, obj):
        """Điểm 'dừng đột ngột' từ lịch sử vận tốc (0..1).

        Va chạm mạnh làm vận tốc nhảy từ cao về ~0 trong 1 frame: lúc đó
        _get_deceleration trả 0 (vì vel≈0). Đặc trưng này bắt chính xác tình
        huống đó: trước đó chạy nhanh (prev_fast), bây giờ gần đứng yên (now).
        """
        if len(obj.velocity_history) < 4:
            return 0.0
        speeds = [np.linalg.norm(v) for v in list(obj.velocity_history)[-4:]]
        now_speed = speeds[-1]
        prev_fast = max(speeds[:-1])
        if now_speed < 2.0 and prev_fast > 10.0:
            return min(1.0, prev_fast / 20.0)
        return 0.0

    def _get_direction_change(self, obj):
        if len(obj.direction_history) < 3:
            return 0.0
        dirs = list(obj.direction_history)
        delta = dirs[-1] - dirs[-3]
        # Normalize angular difference to [0, pi] (handle +/- pi seam)
        return abs(np.arctan2(np.sin(delta), np.cos(delta)))

    def cleanup_lost_tracks(self, active_track_ids):
        """Xoá state cho các track đã mất dấu."""
        lost_pairs = [k for k in self.collision_state
                      if k[0] not in active_track_ids or k[1] not in active_track_ids]
        for k in lost_pairs:
            del self.collision_state[k]

        lost_singles = [k for k in self.hard_stop_state if k not in active_track_ids]
        for k in lost_singles:
            del self.hard_stop_state[k]

        for k in lost_pairs:
            self.pair_dist_hist.pop(k, None)
            self.pair_contact_frame.pop(k, None)

        # Vehicle ↔ Object / Falling object state
        for state_map in (self.object_collision_state, self.object_fall_state):
            lost_obj_pairs = [k for k in state_map
                              if k[0] not in active_track_ids
                              or k[1] not in active_track_ids]
            for k in lost_obj_pairs:
                del state_map[k]

        lost_fall = [k for k in self.object_fall_vy_hist if k not in active_track_ids]
        for k in lost_fall:
            del self.object_fall_vy_hist[k]
