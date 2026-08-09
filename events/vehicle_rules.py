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

        # ------------------------------------------------------------
        # FEATURE 6: Post-contact — dừng/đổi hướng bất thường sau tiếp xúc
        # ------------------------------------------------------------
        self._update_contact_frame(pair_key, is_proximate)
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
            + w["decel"] * decel_score
            + w["post_contact"] * post_contact_score
        )

        # Gate bắt buộc: phải gần nhau (proximity là điều kiện cần)
        is_proximate_candidate = is_proximate and score > config.VEHICLE_COLLISION_SCORE_THRESH

        # Cho phép "hậu va chạm": ngay sau thời điểm tiếp xúc, xe bị hất lệch
        # (đổi hướng đột ngột) hoặc dừng bất thường dù 2 xe đã bắt đầu tách xa.
        # Tránh mất tín hiệu đúng lúc vụ va chạm đang xảy ra.
        post_contact_only = (
            pair_key in self.pair_contact_frame
            and post_contact_score > 0.7
            and score > config.VEHICLE_COLLISION_SCORE_THRESH * 0.8
        )

        is_candidate = is_proximate_candidate or post_contact_only

        # Temporal sustained confirmation (tránh spike 1 frame)
        if is_candidate:
            self.collision_state[pair_key] = self.collision_state.get(pair_key, 0) + 1
        else:
            self.collision_state[pair_key] = max(
                0, self.collision_state.get(pair_key, 0) - 1
            )

        is_confirmed = (self.collision_state.get(pair_key, 0)
                        >= config.VEHICLE_COLLISION_SUSTAINED)
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
        tid = obj.track_id
        if is_candidate:
            self.hard_stop_state[tid] = self.hard_stop_state.get(tid, 0) + 1
        else:
            self.hard_stop_state[tid] = max(0, self.hard_stop_state.get(tid, 0) - 1)

        return self.hard_stop_state.get(tid, 0) >= config.VEHICLE_HARD_STOP_SUSTAINED

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

    def _update_contact_frame(self, pair_key, is_proximate):
        """Ghi nhận frame gần nhất mà 2 xe đạt proximity (thời điểm tiếp xúc)."""
        if is_proximate:
            self.pair_contact_frame[pair_key] = config.VEHICLE_POST_CONTACT_WINDOW
        else:
            if pair_key in self.pair_contact_frame:
                self.pair_contact_frame[pair_key] -= 1
                if self.pair_contact_frame[pair_key] <= 0:
                    del self.pair_contact_frame[pair_key]

    def _get_post_contact_anomaly(self, pair_key, objA, objB,
                                   dir_changeA, dir_changeB):
        """Đặc trưng hậu va chạm: 1 xe dừng / đổi hướng NGAY SAU tiếp xúc.

        Nếu vừa xảy ra tiếp xúc (trong cửa sổ POST_CONTACT_WINDOW) và có dấu
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
