"""False-positive tests cho RESTRICTED_INTRUSION (vùng cấm).

KEY FALSE POSITIVES phải loại:
- Người đứng YÊN trong vùng cấm (không di chuyển) -> KHÔNG xâm nhập
  (rule yêu cầu Movement + Dwell, không chỉ "đứng trong vùng cấm").
- Người chỉ chạm mép polygon (không vào sâu) -> KHÔNG xâm nhập.
Sanity: người đi sâu + lưu tại chỗ -> candidate.
"""
import unittest

import config
from events.intrusion_rules import IntrusionRules
from tracker.object_state import TrackedObjectState

_ZONE = {
    "name": "Restricted",
    "event_type": config.EVENT_TYPE_INTRUSION,
    "polygon": [[0, 0], [640, 0], [640, 384], [0, 384]],
}


class TestIntrusionFalsePositive(unittest.TestCase):
    def setUp(self):
        self.rules = IntrusionRules()

    def test_person_standing_still_deep_inside_not_intrusion(self):
        # đứng YÊN tại tâm vùng cấm 15 frame: depth ok nhưng displacement=0
        obj = TrackedObjectState(1, 0, maxlen=30)
        box = [280, 80, 360, 300]          # person_h = 220, tâm (320,190) sâu
        t0 = 1000.0
        seen = 0
        for i in range(15):
            obj.update(box, t0 + i * 0.2)
            events = self.rules.check_intrusion(obj, [_ZONE])
            seen += len(events)
        self.assertEqual(0, seen,
                         "Đứng yên trong vùng cấm KHÔNG được báo xâm nhập")

    def test_person_brushing_polygon_edge_not_intrusion(self):
        # tâm ở sát mép, depth < INTRUSION_DEPTH_RATIO * person_h
        obj = TrackedObjectState(2, 0, maxlen=30)
        box = [5, 100, 105, 300]           # tâm (55,200) — sát mép trái, h=200
        t0 = 1000.0
        for i in range(15):
            obj.update(box, t0 + i * 0.2)
            events = self.rules.check_intrusion(obj, [_ZONE])
            self.assertEqual([], events, "Chạm mép không phải xâm nhập")


class TestIntrusionSanity(unittest.TestCase):
    def setUp(self):
        self.rules = IntrusionRules()

    def test_moving_person_confirms_candidate(self):
        # người đi sâu vào + di chuyển + lưu lại -> candidate
        obj = TrackedObjectState(3, 0, maxlen=40)
        t0 = 1000.0
        candidate = None
        for i in range(20):
            x = 320 + i * 2                  # di chuyển 2px/frame
            obj.update([x, 80, x + 80, 300], t0 + i * 0.2)
            events = self.rules.check_intrusion(obj, [_ZONE])
            if events:
                candidate = events[0]
        self.assertIsNotNone(candidate, "Đi sâu + di chuyển phải sinh candidate")
        self.assertEqual("RESTRICTED_INTRUSION", candidate["event_type"])
        self.assertGreaterEqual(candidate["confidence"],
                                config.rule_meta("RESTRICTED_INTRUSION")["candidate"])


if __name__ == "__main__":
    unittest.main()