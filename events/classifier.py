from events.person_rules import PersonActionRules
from events.vehicle_rules import VehicleAccidentRules
from events.smoke_fire_rules import SmokeFireRules
from events.intrusion_rules import IntrusionRules
from events.confirm import EventConfirmTracker
import config


class RuleBasedEventClassifier:
    """Classifier tổng hợp gom toàn bộ 4 logic sự kiện.

    Cải tiến:
    - Object-based rules CHỈ chạy trên detection frame (dữ liệu kinematic mới).
    - Visual pattern rules (smoke/fire) chạy MỌI frame (tận dụng temporal analysis).
    - Cleanup lost track state tự động.
    - Tách biệt detection frame vs predicted frame để confirm counter chính xác.
    """

    def __init__(self):
        self.person_rules = PersonActionRules()
        self.vehicle_rules = VehicleAccidentRules()
        self.smoke_fire_rules = SmokeFireRules()
        self.intrusion_rules = IntrusionRules()
        self.confirmer = EventConfirmTracker(
            default_frames=config.EVENT_CONFIRM_FRAMES_DEFAULT
        )

    def evaluate(self, camera_id, memory_manager, frame_bgr,
                 is_detection_frame=True, frame_idx=0):
        """Đánh giá tất cả event types trên frame hiện tại.

        Args:
            camera_id: ID camera.
            memory_manager: ObjectMemoryManager chứa temporal history.
            frame_bgr: Frame BGR gốc.
            is_detection_frame: True nếu frame này có detection thật (không phải predicted).
                Object-based rules chỉ chạy trên detection frame để tránh đếm trùng
                khi confirm. Smoke/Fire chạy mọi frame.
            frame_idx: Số thứ tự frame.
        """
        candidates = []
        rois = config.CAMERA_ROIS.get(camera_id, [])

        # ============================================================
        # Logic 3: Smoke / Fire — chạy MỌI frame
        # Visual pattern analysis hưởng lợi từ flicker detection frame-to-frame.
        # Persistence logic bên trong SmokeFireRules tự quản lý temporal.
        # ============================================================
        candidates.extend(self.smoke_fire_rules.analyze_frame(frame_bgr, rois))

        # ============================================================
        # Object-based rules — CHỈ chạy trên DETECTION frame
        # Trên predicted frame, kinematic data không đổi → đếm trùng confirm counter.
        # Bỏ qua predicted frame để confirm 1:1 với detection thật.
        # ============================================================
        if is_detection_frame:
            objects = list(memory_manager.visible_objects().values())
            active_ids = set(obj.track_id for obj in objects)

            # --------------------------------------------------------
            # Single-Object Analysis
            # --------------------------------------------------------
            for obj in objects:
                # Logic 4: Intrusion (mọi class 0)
                candidates.extend(
                    self.intrusion_rules.check_intrusion(obj, rois)
                )

                # Logic 1: Person fall & gesture (class 0)
                if obj.cls_id == 0:
                    if self.person_rules.check_fall(obj):
                        candidates.append({
                            "event_type": "HUMAN_FALL",
                            "track_ids": [obj.track_id],
                            "confidence": 0.90,
                            "description": "Phát hiện người bị ngã",
                        })


                # Logic 2: Vehicle hard stop (class 2,3,5,7)
                if obj.cls_id in [2, 3, 5, 7]:
                    if self.vehicle_rules.check_hard_stop(obj):
                        candidates.append({
                            "event_type": "VEHICLE_STOP_ANOMALY",
                            "track_ids": [obj.track_id],
                            "confidence": 0.85,
                            "description":
                                "Xe tai nạn / dừng bất thường giữa đường",
                        })

            # --------------------------------------------------------
            # Pairwise Analysis
            # --------------------------------------------------------
            n = len(objects)
            for i in range(n):
                for j in range(i + 1, n):
                    oA, oB = objects[i], objects[j]

                    # Logic 1: Conflict / Fight (2 person)
                    if oA.cls_id == 0 and oB.cls_id == 0:
                        is_conflict, score = self.person_rules.check_conflict(
                            oA, oB
                        )
                        if is_conflict:
                            candidates.append({
                                "event_type": "HUMAN_CONFLICT",
                                "track_ids": [oA.track_id, oB.track_id],
                                "confidence": min(1.0, 0.7 + score / 60.0),
                                "description":
                                    "Phát hiện xô xát/đánh nhau/giằng co",
                            })

                        # Person collision (nếu chưa đủ conflict)
                        is_approach, a_score = (
                            self.person_rules.check_person_collision(oA, oB)
                        )
                        if is_approach and not is_conflict:
                            candidates.append({
                                "event_type": "PERSON_COLLISION",
                                "track_ids": [oA.track_id, oB.track_id],
                                "confidence": a_score,
                                "description":
                                    "Phát hiện 2 người tiếp cận nhanh/va chạm",
                            })

                    # Logic 2: Vehicle collision
                    if oA.cls_id in [2, 3, 5, 7] or oB.cls_id in [2, 3, 5, 7]:
                        if self.vehicle_rules.check_collision(oA, oB):
                            candidates.append({
                                "event_type": "VEHICLE_COLLISION",
                                "track_ids": [oA.track_id, oB.track_id],
                                "confidence": 0.92,
                                "description":
                                    "Phát hiện va chạm phương tiện/vật thể",
                            })

            # --------------------------------------------------------
            # Cleanup lost track state (giải phóng memory, tránh ghost alert)
            # --------------------------------------------------------
            self.vehicle_rules.cleanup_lost_tracks(active_ids)
            self.person_rules.cleanup_lost_tracks(active_ids)

        # Score / Temporal Confirm
        # decay chỉ chạy trên detection frame (decay=False trên predicted frame)
        # để counter object-based tích lũy đúng nhịp 1:1 với detection thật.
        return self.confirmer.process(candidates, decay=is_detection_frame)