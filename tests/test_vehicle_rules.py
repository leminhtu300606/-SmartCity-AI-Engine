"""Rule 6C false-positive tests cho vehicle collision.

KEY FALSE POSITIVES phải loại:
- "0.38 car = plastic object": vehicle conf thấp KHÔNG được vào collision.
- 2 xe ĐỖ CẠNH NHAU (chồng bbox, không động học) -> KHÔNG tai nạn.
- xe đơn điều hướng bình thường -> KHÔNG deform.
"""
import unittest

import config
from events.vehicle_rules import VehicleAccidentRules
from tracker.object_state import TrackedObjectState


def make_vehicle(track_id, boxes):
    obj = TrackedObjectState(track_id, 2, maxlen=config.TEMPORAL_BUFFER_MAXLEN)
    obj.conf = 0.9
    t0 = 1000.0
    for i, b in enumerate(boxes):
        obj.update(list(b), t0 + i * 0.2)
    return obj


class TestCollisionFalsePositive(unittest.TestCase):
    def setUp(self):
        self.rules = VehicleAccidentRules()

    def _approaching_boxes(self, start_x, end_x, n=8):
        """Xe tiến gần theo trục x từ start_x về end_x (giữ nguyên y/h)."""
        boxes = []
        for i in range(n):
            x = start_x + (end_x - start_x) * i / (n - 1)
            boxes.append([x, 50, x + 100, 150])
        return boxes

    def test_low_confidence_vehicle_rejected(self):
        # "0.38 car = plastic object": conf < gate -> thẳng bị loại.
        objA = make_vehicle(1, [[0, 50, 100, 150]] * 8)
        objB = make_vehicle(2, self._approaching_boxes(400, 80, 8))
        objA.conf = 0.38          # plastic object / detection mơ hồ
        objB.conf = 0.90
        cand = self.rules.eval_collision(objA, objB)
        self.assertIsNone(cand,
                          "Vehicle conf < VEHICLE_COLLISION_MIN_CONF bị REJECT")

    def test_parked_adjacent_vehicles_not_collision(self):
        # 2 xe ĐỖ sát nhau, chồng bbox, nhưng KHÔNG động học -> không tai nạn
        boxA = [0, 50, 100, 150]
        boxB = [80, 50, 180, 150]
        objA = make_vehicle(1, [boxA] * 8)
        objB = make_vehicle(2, [boxB] * 8)
        for _ in range(6):
            cand = self.rules.eval_collision(objA, objB)
            if cand is not None:
                self.fail("2 xe đỗ cạnh nhau KHÔNG được báo tai nạn")
        self.assertEqual(
            self.rules.collision_state.get((1, 2), 0), 0)

    def test_parallel_moving_vehicles_not_collision(self):
        # 2 xe đi song song cùng hướng (không closing) -> KHÔNG tai nạn
        shiftA = [[x, 50, x + 100, 150] for x in range(0, 160, 20)]
        objA = make_vehicle(1, shiftA)
        shiftB = [[x + 30, 50, x + 130, 150] for x in range(0, 160, 20)]
        objB = make_vehicle(2, shiftB)
        cand = self.rules.eval_collision(objA, objB)
        self.assertIsNone(cand)

    def test_deformation_normal_driving_ignored(self):
        # xe đi thẳng/đều -> không deform
        obj = make_vehicle(3, [[x, 50, x + 100, 150] for x in range(0, 240, 20)])
        cand = self.rules.eval_deformation(obj, frame_size=config.MODEL_INPUT_SIZE)
        self.assertIsNone(cand)


class TestCollisionSanity(unittest.TestCase):
    """Positive control: xe lao vào nhau -> candidate (cần gate conf cao)."""

    def setUp(self):
        self.rules = VehicleAccidentRules()

    def test_closing_vehicles_emit_candidate(self):
        objA = make_vehicle(1, [[0, 50, 100, 150]] * 12)
        # B lao thẳng về phía A, dừng chồng lên A ở frame cuối
        objB = make_vehicle(2, [[x, 50, x + 100, 150]
                                for x in range(340, 20, -30)])
        objA.conf = 0.9
        objB.conf = 0.9
        cand = self.rules.eval_collision(objA, objB)
        self.assertIsNotNone(cand, "2 xe lao vào nhau phải sinh candidate")
        self.assertEqual("VEHICLE_COLLISION", cand["event_type"])
        self.assertGreaterEqual(cand["confidence"],
                                config.COLLISION_CANDIDATE_THRESH)


if __name__ == "__main__":
    unittest.main()