from events.person_rules import PersonActionRules
from events.vehicle_rules import VehicleAccidentRules
from events.smoke_fire_rules import SmokeFireRules
from events.intrusion_rules import IntrusionRules
from events.confirm import EventConfirmTracker
import config


_VEHICLE_NAMES = {
    1: "xe đạp",
    2: "ô tô",
    3: "xe máy",
    5: "xe buýt",
    6: "tàu hỏa",
    7: "xe tải",
}


def name_of(obj):
    vtype = getattr(obj, "vehicle_type", None)
    if vtype:
        return vtype
    return _VEHICLE_NAMES.get(obj.cls_id, f"phương tiện #{obj.cls_id}")


class RuleBasedEventClassifier:
    """Classifier tổng hợp theo pipeline Phase 1 (4 nhóm event).

    Pipeline:
      ByteTrack / OC-SORT  →  Temporal State / History  →  Pose Analysis +
      Motion Analysis  →  Rule-based Classifier  →  Event Score / Confirm
      →  Alert + Metadata.

    Phase 1 chỉ tập trung 4 event:
      1. Person fall / conflict   (HUMAN_FALL, HUMAN_CONFLICT)
      2. Vehicle collision        (VEHICLE_COLLISION — xe-xe)
      3. Smoke / Fire             (FIRE_DETECTED)
      4. Restricted-zone intrusion(RESTRICTED_INTRUSION)

    - Object-based rules CHỈ chạy trên detection frame (dữ liệu kinematic mới).
    - Visual pattern rules (smoke/fire) chạy MỌI frame (tận dụng temporal analysis).
    - Cleanup lost track state tự động.
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
            frame_idx: Số thứ tự frame.
        """
        candidates = []
        rois = config.CAMERA_ROIS.get(camera_id, [])

        # ============================================================
        # Smoke / Fire — chạy MỌI frame
        # Visual pattern analysis hưởng lợi từ flicker detection frame-to-frame.
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
        # ============================================================
        if is_detection_frame:
            objects = list(memory_manager.visible_objects().values())
            active_ids = set(obj.track_id for obj in objects)
            persons = [obj for obj in objects if obj.cls_id == 0]

            # --------------------------------------------------------
            # Single-Object Analysis (person)
            # --------------------------------------------------------
            for obj in objects:
                # Intrusion (chỉ person)
                candidates.extend(
                    self.intrusion_rules.check_intrusion(obj, rois)
                )

                # Person fall (chỉ person)
                if obj.cls_id == 0 and self.person_rules.check_fall(obj):
                    candidates.append({
                        "event_type": "HUMAN_FALL",
                        "track_ids": [obj.track_id],
                        "bbox": self._get_bbox(obj),
                        "confidence": 0.90,
                        "description": "Phát hiện người bị ngã",
                        "evidence_objects": [self._object_evidence(obj)],
                    })

                # Xe ĐƠN bị BIẾN DẠNG (thay đổi so với ban đầu) = tai nạn.
                # Tín hiệu độc lập: không cần cặp xe thứ 2 — bắt xe lật/đâm vật
                # cản/va vào đối tượng không được track. Đã loại trừ che khuất
                # (màn hình cắt / vật thể khác che).
                if obj.cls_id in config.VEHICLE_CLASSES:
                    other_bboxes = [
                        o.bbox_history[-1]
                        for o in objects
                        if o.track_id != obj.track_id
                    ]
                    is_deformed, deform_score = \
                        self.vehicle_rules.check_deformation(
                            obj, other_bboxes=other_bboxes,
                            frame_size=config.MODEL_INPUT_SIZE)
                    if is_deformed:
                        candidates.append({
                            "event_type": "VEHICLE_COLLISION",
                            "track_ids": [obj.track_id],
                            "bbox": self._get_bbox(obj),
                            "confidence": min(0.93, 0.80 + deform_score * 0.15),
                            "description":
                                f"[VA CHẠM GIAO THÔNG] {name_of(obj)} bị "
                                f"biến dạng đột ngột (thay đổi so với ban đầu)",
                            "evidence_objects": [self._object_evidence(obj)],
                        })

            # --------------------------------------------------------
            # Person Interaction — XÔ XÁT / ĐÁNH NHAU (cặp người)
            # Dụng cụ (TOOL): object không phải người và không phải phương tiện
            # (chai, gậy, ghế, túi...) chuyển động nhanh gần người → kênh tầm xa.
            # --------------------------------------------------------
            tool_objects = [
                obj for obj in objects
                if obj.cls_id != 0 and obj.cls_id not in config.VEHICLE_CLASSES
            ]
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    oA, oB = persons[i], persons[j]
                    centA = oA.center_history[-1]
                    zone_a = config.grid_zone(centA[0], centA[1])
                    is_conflict, score = self.person_rules.check_conflict(
                        oA, oB, tool_objects=tool_objects)
                    if is_conflict:
                        candidates.append({
                            "event_type": "HUMAN_CONFLICT",
                            "track_ids": [oA.track_id, oB.track_id],
                            "bbox": self._union_bbox(oA, oB),
                            "confidence": min(1.0, 0.80 + score / 40.0),
                            "description":
                                "Phát hiện xô xát/đánh nhau "
                                "(gồm cả đánh võ/vật biểu diễn)",
                            "zone_name": f"grid_{zone_a}",
                            "evidence_objects": [
                                self._object_evidence(oA),
                                self._object_evidence(oB),
                            ],
                        })

            # --------------------------------------------------------
            # Pairwise Analysis — VEHICLE COLLISION (xe-xe)
            # --------------------------------------------------------
            vehicles = [obj for obj in objects if obj.cls_id in config.VEHICLE_CLASSES]
            frame_size = config.MODEL_INPUT_SIZE
            for i in range(len(vehicles)):
                for j in range(i + 1, len(vehicles)):
                    oA, oB = vehicles[i], vehicles[j]
                    # Bbox các vật thể KHÁC (ngoài cặp) để loại trừ che khuất —
                    # xe bị vật khác che → bbox méo nhưng không phải biến dạng
                    # do tai nạn (chống false positive từ occlusion).
                    other_bboxes = [
                        o.bbox_history[-1]
                        for o in objects
                        if o.track_id != oA.track_id and o.track_id != oB.track_id
                    ]
                    is_collision, c_score = self.vehicle_rules.check_collision(
                        oA, oB, other_bboxes=other_bboxes, frame_size=frame_size)
                    if is_collision:
                        candidates.append({
                            "event_type": "VEHICLE_COLLISION",
                            "track_ids": [oA.track_id, oB.track_id],
                            "bbox": self._union_bbox(oA, oB),
                            "confidence": min(0.98, 0.75 + c_score * 0.25),
                            "description":
                                f"[VA CHẠM GIAO THÔNG] {name_of(oA)} và "
                                f"{name_of(oB)} va chạm",
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
