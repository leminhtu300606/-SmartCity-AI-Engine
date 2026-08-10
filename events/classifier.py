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
        # Truyền bbox NGƯỜI (cls 0) để loại áo quần đỏ/cam (không phải lửa).
        # ============================================================
        person_bboxes = [
            obj.predicted_bbox if obj.predicted_bbox is not None
            else obj.bbox_history[-1]
            for obj in memory_manager.visible_objects().values()
            if obj.cls_id == 0
        ]
        candidates.extend(
            self.smoke_fire_rules.analyze_frame(
                frame_bgr, rois, object_bboxes=person_bboxes)
        )

        # ============================================================
        # Object-based rules — CHỈ chạy trên DETECTION frame
        # Trên predicted frame, kinematic data không đổi → đếm trùng confirm counter.
        # Bỏ qua predicted frame để confirm 1:1 với detection thật.
        # ============================================================
        if is_detection_frame:
            objects = list(memory_manager.visible_objects().values())
            active_ids = set(obj.track_id for obj in objects)

            # ============================================================
            # BƯỚC 0: XÁC ĐỊNH RÕ SỐ LƯỢNG NGƯỜI TRƯỚC
            # Số lượng người quyết định các bước phân tích NGƯỜI phía sau:
            #   - 1 người   : chỉ phân tích ĐƠN (ngã) — không có tương tác.
            #   - 2 người   : phân tích theo CẶP (xô xát, va chạm người).
            #   - >= 3 người: thêm phân tích CỤM (xô xát nhóm nhiều người).
            # ============================================================
            persons = [obj for obj in objects if obj.cls_id == 0]
            num_persons = len(persons)

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
                            "bbox": self._get_bbox(obj),
                            "confidence": 0.90,
                            "description": "Phát hiện người bị ngã",
                            "evidence_objects": [self._object_evidence(obj)],
                        })


                # Logic 2: Vehicle hard stop (class 2,3,5,7)
                if obj.cls_id in [2, 3, 5, 7]:
                    if self.vehicle_rules.check_hard_stop(obj):
                        candidates.append({
                            "event_type": "VEHICLE_STOP_ANOMALY",
                            "track_ids": [obj.track_id],
                            "bbox": self._get_bbox(obj),
                            "confidence": 0.90,
                            "description":
                                "Xe tai nạn / dừng bất thường giữa đường",
                                "evidence_objects": [self._object_evidence(obj)],
                        })

                    # Stage 1: xác định TÌNH TRẠNG xe — nghiêng/lật (tai nạn 1 xe)
                    is_tilted, tilt_score = self.vehicle_rules.check_vehicle_state(obj)
                    if is_tilted:
                        candidates.append({
                            "event_type": "VEHICLE_ACCIDENT",
                            "track_ids": [obj.track_id],
                            "bbox": self._get_bbox(obj),
                            "confidence": min(0.95, tilt_score),
                            "description":
                                "Phát hiện xe nghiêng/lật (nghi tai nạn)",
                        })

            # --------------------------------------------------------
            # Person Interaction — phân nhánh THEO SỐ LƯỢNG NGƯỜI
            # BƯỚC 0 đã đếm num_persons; chỉ chạy đúng bộ phân tích tương ứng.
            # --------------------------------------------------------

            # --- Số người >= 2: phân tích theo CẶP người-người ---
            # (xô xát/đánh nhau + va chạm giữa người)
            if num_persons >= 2:
                for i in range(num_persons):
                    for j in range(i + 1, num_persons):
                        oA, oB = persons[i], persons[j]

                        # Logic 1: Conflict / Fight (2 person)
                        centA = oA.center_history[-1]
                        zone_a = config.grid_zone(centA[0], centA[1])
                        is_conflict, score = self.person_rules.check_conflict(
                            oA, oB
                        )
                        if is_conflict:
                            candidates.append({
                                "event_type": "HUMAN_CONFLICT",
                                "track_ids": [oA.track_id, oB.track_id],
                                "bbox": self._union_bbox(oA, oB),
                                "confidence": min(1.0, 0.80 + score / 40.0),
                                "description":
                                    "Phát hiện xô xát/đánh nhau/giằng co",
                                    "zone_name": f"grid_{zone_a}",
                                    "evidence_objects": [
                                        self._object_evidence(oA),
                                        self._object_evidence(oB),
                                    ],
                            })

                        # Person collision (nếu chưa đủ conflict)
                        is_approach, a_score = (
                            self.person_rules.check_person_collision(oA, oB)
                        )
                        if is_approach and not is_conflict:
                            candidates.append({
                                "event_type": "PERSON_COLLISION",
                                "track_ids": [oA.track_id, oB.track_id],
                                "bbox": self._union_bbox(oA, oB),
                                "confidence": a_score,
                                "description":
                                    "Phát hiện 2 người tiếp cận nhanh/va chạm",
                                "zone_name": f"grid_{zone_a}",
                                "evidence_objects": [
                                    self._object_evidence(oA),
                                    self._object_evidence(oB),
                                ],
                            })

            # --- Số người >= 3: thêm phân tích CỤM người ---
            # gom cụm 3 người khi lịch sử ngắn cho thấy họ cùng dính sát
            # và có ít nhất một người bất thường.
            # Phân nhóm theo Ô LƯỚI (grid zone): chỉ gom cụm những người
            # CÙNG vùng — người ở ô khác không tương tác với nhau.
            if num_persons >= 3:
                zone_person_map = {}
                for obj in persons:
                    cent = obj.center_history[-1]
                    z = config.grid_zone(cent[0], cent[1])
                    zone_person_map.setdefault(z, []).append(obj)

                for z, zone_persons in zone_person_map.items():
                    if len(zone_persons) < 3:
                        continue
                    group_confirmed, group_score, group_ids = (
                        self.person_rules.check_group_conflict(zone_persons)
                    )
                    if group_confirmed and group_ids:
                        group_objs = [
                            obj for obj in zone_persons
                            if obj.track_id in group_ids
                        ]
                        if group_objs:
                            candidates.append({
                                "event_type": config.EVENT_TYPE_HUMAN_GROUP_CONFLICT,
                                "track_ids": list(group_ids),
                                "bbox": self._union_many_bboxes(group_objs),
                                "confidence": min(0.96, group_score),
                                "description":
                                    "Phát hiện cụm 3 người tiếp cận/va chạm/xô xát",
                                "zone_name": f"grid_{z}",
                                "evidence_objects": [
                                    self._object_evidence(obj)
                                    for obj in group_objs
                                ],
                            })

            # --------------------------------------------------------
            # Pairwise Analysis — VEHICLE
            # Không phụ thuộc số lượng người (xe va chạm xe / vật thể / người)
            # --------------------------------------------------------
            n = len(objects)
            for i in range(n):
                for j in range(i + 1, n):
                    oA, oB = objects[i], objects[j]

                    # Logic 2: Vehicle collision
                    if oA.cls_id in [2, 3, 5, 7] or oB.cls_id in [2, 3, 5, 7]:
                        # 2.1) Xe - Xe (cả 2 là phương tiện)
                        if oA.cls_id in [2, 3, 5, 7] and oB.cls_id in [2, 3, 5, 7]:
                            is_collision, c_score = self.vehicle_rules.check_collision(oA, oB)
                            if is_collision:
                                candidates.append({
                                    "event_type": "VEHICLE_COLLISION",
                                    "track_ids": [oA.track_id, oB.track_id],
                                    "bbox": self._union_bbox(oA, oB),
                                    "confidence": min(0.98, 0.75 + c_score * 0.25),
                                    "description":
                                        "Phát hiện va chạm phương tiện/vật thể",
                                    "evidence_objects": [
                                        self._object_evidence(oA),
                                        self._object_evidence(oB),
                                    ],
                                })

                        # 2.2) Xe - Vật thể / người (chỉ 1 bên là xe)
                        else:
                            objV = oA if oA.cls_id in [2, 3, 5, 7] else oB
                            objO = oB if objV is oA else oA
                            is_obj_collision, oc_score = (
                                self.vehicle_rules.check_object_collision(objV, objO)
                            )
                            if is_obj_collision:
                                candidates.append({
                                    "event_type": config.EVENT_TYPE_VEHICLE_OBJECT_COLLISION,
                                    "track_ids": [oA.track_id, oB.track_id],
                                    "bbox": self._union_bbox(oA, oB),
                                    "confidence": min(0.95, 0.70 + oc_score * 0.25),
                                    "description":
                                        "Phát hiện xe va chạm vật thể/người",
                                    "evidence_objects": [
                                        self._object_evidence(oA),
                                        self._object_evidence(oB),
                                    ],
                                })

                            # Stage 1: xác định TÌNH TRẠNG xe bị ĐÈ/va phải mạnh
                            is_crushed, crush_score = (
                                self.vehicle_rules.check_crushed(objV, objO)
                            )
                            if is_crushed:
                                candidates.append({
                                    "event_type": config.EVENT_TYPE_VEHICLE_OBJECT_COLLISION,
                                    "track_ids": [oA.track_id, oB.track_id],
                                    "bbox": self._union_bbox(oA, oB),
                                    "confidence": min(0.95, crush_score),
                                    "description":
                                        "Phát hiện xe bị vật thể/người đè lên "
                                        "(nghi tai nạn)",
                                    "evidence_objects": [
                                        self._object_evidence(oA),
                                        self._object_evidence(oB),
                                    ],
                                })

                        # 2.3) Vật thể rơi vào xe (cả 2 hướng: xe-vật, vật-xe)
                        is_falling, f_score = (
                            self.vehicle_rules.check_object_falling(oA, oB)
                        )
                        if is_falling:
                            candidates.append({
                                "event_type": config.EVENT_TYPE_OBJECT_FALLING,
                                "track_ids": [oA.track_id, oB.track_id],
                                "bbox": self._union_bbox(oA, oB),
                                "confidence": f_score,
                                "description":
                                    "Phát hiện vật thể rơi từ trên xuống trúng xe",
                                "evidence_objects": [
                                    self._object_evidence(oA),
                                    self._object_evidence(oB),
                                ],
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

    # ----------------------------------------------------------------
    # Helpers — lấy bbox vị trí event để hiển thị
    # ----------------------------------------------------------------
    @staticmethod
    def _get_bbox(obj):
        """Bbox [x1, y1, x2, y2] hiện tại của object; None nếu chưa có."""
        if len(obj.bbox_history) == 0:
            return None
        box = obj.bbox_history[-1]
        try:
            return [int(v) for v in box]
        except (TypeError, ValueError):
            return None

    @classmethod
    def _union_bbox(cls, objA, objB):
        """Bbox hợp (bao quanh) 2 object — vùng xảy ra sự kiện."""
        bA, bB = cls._get_bbox(objA), cls._get_bbox(objB)
        if bA is None:
            return bB
        if bB is None:
            return bA
        return [
            min(bA[0], bB[0]),
            min(bA[1], bB[1]),
            max(bA[2], bB[2]),
            max(bA[3], bB[3]),
        ]

    @classmethod
    def _union_many_bboxes(cls, objects):
        boxes = [cls._get_bbox(obj) for obj in objects]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            return None
        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    @staticmethod
    def _object_evidence(obj):
        bbox = RuleBasedEventClassifier._get_bbox(obj)
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