from events.person_rules import PersonActionRules
from events.vehicle_rules import VehicleAccidentRules
from events.smoke_fire_rules import SmokeFireRules
from events.intrusion_rules import IntrusionRules
from events.confirm import EventConfirmTracker
import config

class RuleBasedEventClassifier:
    """Classifier tổng hợp gom toàn bộ 4 logic sự kiện."""
    def __init__(self):
        self.person_rules = PersonActionRules()
        self.vehicle_rules = VehicleAccidentRules()
        self.smoke_fire_rules = SmokeFireRules()
        self.intrusion_rules = IntrusionRules()
        self.confirmer = EventConfirmTracker(confirm_frames=config.EVENT_CONFIRM_FRAMES)

    def evaluate(self, camera_id, memory_manager, frame_bgr):
        candidates = []
        objects = list(memory_manager.objects.values())
        rois = config.CAMERA_ROIS.get(camera_id, [])

        # Logic 3: Smoke / Fire
        candidates.extend(self.smoke_fire_rules.analyze_frame(frame_bgr, rois))

        # Logic 4 & Logic 1 (Single Object)
        for obj in objects:
            candidates.extend(self.intrusion_rules.check_intrusion(obj, rois))
            
            if obj.cls_id == 0 and self.person_rules.check_fall(obj):
                candidates.append({
                    "event_type": "HUMAN_FALL",
                    "track_ids": [obj.track_id],
                    "confidence": 0.90,
                    "description": "Phát hiện người bị ngã"
                })

            if obj.cls_id == 0 and self.person_rules.check_wild_gesture(obj):
                candidates.append({
                    "event_type": "HUMAN_WILD_GESTURE",
                    "track_ids": [obj.track_id],
                    "confidence": 0.80,
                    "description": "Phát hiện vung tay mạnh / hành động bất thường"
                })
            
            if obj.cls_id in [2, 3, 5, 7] and self.vehicle_rules.check_hard_stop(obj):
                candidates.append({
                    "event_type": "VEHICLE_STOP_ANOMALY",
                    "track_ids": [obj.track_id],
                    "confidence": 0.85,
                    "description": "Xe tai nạn / dừng bất thường giữa đường"
                })

        # Logic 1 & Logic 2 (Multi-Object Pairwise)
        n = len(objects)
        for i in range(n):
            for j in range(i + 1, n):
                oA, oB = objects[i], objects[j]

                # Conflict/Fight
                if oA.cls_id == 0 and oB.cls_id == 0:
                    is_conflict, score = self.person_rules.check_conflict(oA, oB)
                    if is_conflict:
                        candidates.append({
                            "event_type": "HUMAN_CONFLICT",
                            "track_ids": [oA.track_id, oB.track_id],
                            "confidence": min(1.0, score / 30.0),
                            "description": "Phát hiện xô xát/đánh nhau/giằng co"
                        })

                    # 2 người tiếp cận nhanh / va chạm (nếu chưa đủ mức conflict)
                    is_approach, a_score = self.person_rules.check_person_collision(oA, oB)
                    if is_approach and not is_conflict:
                        candidates.append({
                            "event_type": "PERSON_COLLISION",
                            "track_ids": [oA.track_id, oB.track_id],
                            "confidence": a_score,
                            "description": "Phát hiện 2 người tiếp cận nhanh/va chạm"
                        })

                # Vehicle Collision
                if oA.cls_id in [2, 3, 5, 7] or oB.cls_id in [2, 3, 5, 7]:
                    if self.vehicle_rules.check_collision(oA, oB):
                        candidates.append({
                            "event_type": "VEHICLE_COLLISION",
                            "track_ids": [oA.track_id, oB.track_id],
                            "confidence": 0.92,
                            "description": "Phát hiện va chạm phương tiện/vật thể"
                        })

        # Score / Temporal Confirm
        return self.confirmer.process(candidates)