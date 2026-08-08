from events.person_rules import PersonActionRules
from events.vehicle_rules import VehicleAccidentRules
from events.intrusion_rules import IntrusionRules

class RuleBasedEventClassifier:
    """Điều phối toàn bộ rule-classifier cho các nhóm sự kiện phase 1."""

    def __init__(self, fps=25):
        self.dt = 1.0 / fps
        self.person_rules = PersonActionRules(self.dt)
        self.vehicle_rules = VehicleAccidentRules(self.dt)
        self.intrusion_rules = IntrusionRules()

    def evaluate_events(self, camera_id, memory_manager):
        detected_events = []
        objects = memory_manager.objects

        # 1. Person: Fall + Intrusion
        for t_id, obj in objects.items():
            if obj.cls_id != 0:
                continue

            intrusion_zones = self.intrusion_rules.check(camera_id, obj)
            for zone in intrusion_zones:
                detected_events.append({
                    "event_type": "ZONE_INTRUSION",
                    "track_ids": [int(t_id)],
                    "confidence": 0.95,
                    "zone_name": zone["zone_name"],
                    "description": f"Xâm nhập khu vực cấm {zone['zone_name']}",
                })

            if self.person_rules.check_fall(obj):
                detected_events.append({
                    "event_type": "HUMAN_FALL",
                    "track_ids": [int(t_id)],
                    "confidence": 0.90,
                    "description": "Phát hiện người bị ngã",
                })

        # 2. Conflict giữa các cặp Person
        person_ids = [t_id for t_id, obj in objects.items() if obj.cls_id == 0]
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                idA, idB = person_ids[i], person_ids[j]
                is_fight, score = self.person_rules.check_conflict(objects[idA], objects[idB])
                if is_fight:
                    detected_events.append({
                        "event_type": "HUMAN_CONFLICT",
                        "track_ids": [int(idA), int(idB)],
                        "confidence": float(score),
                        "description": f"Phát hiện xô xát/đánh nhau (Score: {score:.2f})",
                    })

        # 3. Vehicle Accident giữa các cặp phương tiện
        vehicle_ids = [
            t_id for t_id, obj in objects.items()
            if obj.cls_id in VehicleAccidentRules.VEHICLE_CLASS_IDS
        ]
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                idA, idB = vehicle_ids[i], vehicle_ids[j]
                if self.vehicle_rules.check_collision(objects[idA], objects[idB]):
                    detected_events.append({
                        "event_type": "VEHICLE_ACCIDENT",
                        "track_ids": [int(idA), int(idB)],
                        "confidence": 0.88,
                        "description": "Xảy ra va chạm giao thông giữa các phương tiện",
                    })

        return detected_events