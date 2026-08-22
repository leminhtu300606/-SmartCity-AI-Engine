from typing import Any

from events.confirm import EventConfirmTracker
from events.intrusion_rules import IntrusionRules
from events.person_rules import PersonActionRules
from events.smoke_fire_rules import SmokeFireRules
from events.vehicle_rules import VehicleAccidentRules
from tracker.memory_manager import ObjectMemoryManager
import config


class RuleBasedEventClassifier:
    """Classifier tổng hợp theo CASCADE (Rule 4) + Candidate/Confirm (Rule 7).

    LEVEL 0: cheap processing           (memory update — do scheduler)
    LEVEL 1: object detection (YOLO)     (do scheduler, 5 FPS, shared model)
    LEVEL 2: candidate rules             (evaluate())
    LEVEL 3: specialized (pose/veh-cls/fire) — chạy khi có candidate phù hợp
    LEVEL 4: temporal confirmation       (EventConfirmTracker) -> CONFIRMED
    EVENT

    Trả về list event đã CONFIRMED (stage=CONFIRMED). Chỉ controller
    (scheduler/main) đưa CONFIRMED vào alert/snapshot.
    """

    def __init__(self, camera_id: str | None = None):
        self.camera_id = camera_id
        self.person_rules = PersonActionRules()
        self.vehicle_rules = VehicleAccidentRules()
        self.smoke_fire_rules = SmokeFireRules()
        self.intrusion_rules = IntrusionRules()
        self.confirmer = EventConfirmTracker()

    def evaluate(self, frame_bgr: Any, memory_manager: ObjectMemoryManager,
                  frame_idx: int = 0, run_fire: bool = False,
                  run_vehicle: bool = False, run_detect: bool = True,
                  camera_id: str | None = None) -> list:
        """Cascade LEVEL 2+4 trên frame hiện tại -> list CONFIRMED events.

        Args:
            run_detect: True khi tick detect (DETECT_FPS) đến hạn. Object-based
                rules (fall/fight/intrusion) chỉ chạy khi có detection MỚI.
            run_fire: True khi tick fire (FIRE_FPS) đến hạn. Fire/Smoke là
                model chuyên biệt — KHÔNG chạy mọi frame.
            run_vehicle: True khi tick accident (ACCIDENT_FPS) đến hạn.
            camera_id: ID camera (dùng cho vehicle_cls vote ngăn camera).
        """
        ccid = camera_id or self.camera_id
        candidates = []
        rois = config.CAMERA_ROIS.get(ccid, [])
        objects = list(memory_manager.visible_objects().values())
        active_ids = set(o.track_id for o in objects)
        persons = [o for o in objects if o.cls_id == 0]

        person_bboxes = [
            obj.bbox_history[-1] if not obj.last_update_predicted else obj.predicted_bbox
            for obj in objects if obj.cls_id == 0 and obj.bbox_history
        ]

        # ============================================================
        # LEVEL 2 — candidate rules
        # ============================================================
        # Smoke/Fire: chạy TẠI tick fire (FIRE_FPS) không phải mọi frame (Rule 4)
        if run_fire:
            candidates.extend(
                self.smoke_fire_rules.analyze_frame(
                    frame_bgr, rois, object_bboxes=person_bboxes))

        # Object-based rules — chỉ chạy trên DETECTION frame (Rule 3/4).
        # Tick fire/accident đến hạn giữa 2 tick detect: KHÔNG chạy lại
        # fall/fight/intrusion trên dữ liệu cũ (tiết kiệm CPU).
        if run_detect:
            # Fall + Intrusion (single object)
            for obj in objects:
                candidates.extend(self.intrusion_rules.check_intrusion(obj, rois))
                if obj.cls_id == 0:
                    cand = self.person_rules.eval_fall(obj)
                    if cand is not None:
                        candidates.append(cand)

            # Fight (cặp người)
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    cand = self.person_rules.eval_conflict(
                        persons[i], persons[j])
                    if cand is not None:
                        candidates.append(cand)

        # Vehicle collision + deformation: chạy TẠI tick accident (Rule 4)
        if run_vehicle:
            vehicles = [o for o in objects if o.cls_id in config.VEHICLE_CLASSES]
            frame_size = config.MODEL_INPUT_SIZE
            for i in range(len(vehicles)):
                # Xe ĐƠN biến dạng
                cand = self.vehicle_rules.eval_deformation(
                    vehicles[i], other_bboxes=[
                        o.bbox_history[-1] for o in objects
                        if o.track_id != vehicles[i].track_id],
                    frame_size=frame_size)
                if cand is not None:
                    candidates.append(cand)
                for j in range(i + 1, len(vehicles)):
                    cand = self.vehicle_rules.eval_collision(
                        vehicles[i], vehicles[j], other_bboxes=[
                            o.bbox_history[-1] for o in objects
                            if o.track_id != vehicles[i].track_id
                            and o.track_id != vehicles[j].track_id],
                        frame_size=frame_size)
                    if cand is not None:
                        candidates.append(cand)

        # Không để accumulated state (conflict/collision) của track đã mất
        self.vehicle_rules.cleanup_lost_tracks(active_ids)
        self.person_rules.cleanup_lost_tracks(active_ids)

        # ============================================================
        # LEVEL 4 — temporal confirmation (candidate -> confirmed)
        # ============================================================
        confirmed = self.confirmer.process(candidates, decay=True)

        if getattr(config, "AI_DEBUG_CANDIDATES", False):
            for c in candidates:
                print(f"[AI:{ccid}] CANDIDATE {c['event_type']} score={c.get('confidence')} "
                      f"tracks={c.get('track_ids')} zone={c.get('zone_name')}")
        return confirmed