"""Rule 6A/6B false-positive tests cho person rules (ngã + xô xát).

KEY FALSE POSITIVES phải loại:
- 2 người đứng CẠNH NHAU yên tĩnh -> KHÔNG fight.
- 2 người ở 2 khu vực xa -> KHÔNG fight.
- người đứng thẳng -> KHÔNG fall.
- người nằm YÊN 1 frame -> KHÔNG fall confirm (cần persistence).
"""
import math
import unittest

import config
from events.confirm import EventConfirmTracker
from events.person_rules import PersonActionRules
from tracker.object_state import TrackedObjectState


def make_obj(track_id, boxes):
    obj = TrackedObjectState(track_id, 0, maxlen=config.TEMPORAL_BUFFER_MAXLEN)
    obj.conf = 0.9
    t0 = 1000.0
    for i, b in enumerate(boxes):
        obj.update(list(b), t0 + i * 0.2)
    return obj


class TestFallFalsePositive(unittest.TestCase):
    def setUp(self):
        self.rules = PersonActionRules()

    def test_upright_person_not_fall(self):
        # người đứng thẳng suốt -> aspect thấp -> không bao giờ candidate
        boxes = [[100, 100, 200, 300]] * 16
        obj = make_obj(1, boxes)
        for _ in range(16):
            cand = self.rules.eval_fall(obj)
            if cand is not None:
                self.fail("Người đứng thẳng KHÔNG được là FALL")
        self.assertLessEqual(obj.fall_persist_count, 1)

    def test_transient_horizontal_not_confirmed(self):
        # Người nằm ngang THOÁNG QUA rồi đứng dậy -> không báo fall.
        # FallScore cao nhanh (posture/vertical), nhưng CONFIRM cần temporal
        confirmer = EventConfirmTracker()
        obj = make_obj(2, [[100, 100, 200, 300]] * 8)
        horiz = [[100, 240, 320, 275]] * 2   # chỉ 2 frame rồi hết
        wrong_confirm = False
        for i, b in enumerate(horiz):
            obj.update(b, 1000 + (8 + i) * 0.2)
            cand = self.rules.eval_fall(obj)
            confirmed = confirmer.process(
                [cand] if cand else [], decay=True)
            wrong_confirm = wrong_confirm or any(
                e.get("stage") == config.STAGE_CONFIRMED
                for e in confirmed)
        self.assertFalse(wrong_confirm,
                         "Nằm ngang 2 frame KHÔNG được CONFIRMED")

    def test_sustained_fall_confirms(self):
        # Nằm ngang duy trì đủ frame + score >= confirm -> CONFIRMED.
        confirmer = EventConfirmTracker()
        obj = make_obj(3, [[100, 100, 200, 300]] * 8)
        horiz = [[100, 240, 320, 275]] * config.FALL_CONFIRM_FRAMES
        confirmed_seen = False
        for i, b in enumerate(horiz):
            obj.update(b, 1000 + (8 + i) * 0.2)
            cand = self.rules.eval_fall(obj)
            confirmed = confirmer.process(
                [cand] if cand else [], decay=True)
            if confirmed:
                confirmed_seen = any(
                    e.get("stage") == config.STAGE_CONFIRMED
                    and e.get("event_type") == "HUMAN_FALL"
                    for e in confirmed)
        self.assertTrue(confirmed_seen)


class TestFightFalsePositive(unittest.TestCase):
    def setUp(self):
        self.rules = PersonActionRules()

    def test_two_people_standing_close_not_fight(self):
        # 2 người đứng CẠNH NHAU, thể chất gần nhau nhưng KHÔNG giật cục
        boxA = [200, 100, 300, 300]
        boxB = [280, 100, 380, 300]
        objA = make_obj(1, [boxA] * 12)
        objB = make_obj(2, [boxB] * 12)
        cand = self.rules.eval_conflict(objA, objB)
        self.assertIsNone(cand)  # "2 người đứng cạnh nhau -> 0"

    def test_two_people_far_apart_not_fight(self):
        objA = make_obj(1, [[20, 100, 120, 300]] * 12)
        objB = make_obj(2, [[520, 100, 620, 300]] * 12)
        cand = self.rules.eval_conflict(objA, objB)
        self.assertIsNone(cand)  # quá xa -> không tương tác

    def test_people_walking_together_not_fight(self):
        # đi sát nhau, cùng vận tốc, không giật cục -> NOT fight
        shift = [(200 + i * 15, 100, 300 + i * 15, 300) for i in range(12)]
        objA = make_obj(1, shift)
        objB = make_obj(2, [(x + 40, y, x2 + 40, y2)
                            for x, y, x2, y2 in shift])
        objA2 = make_obj(3, shift)
        self.assertIsNone(self.rules.eval_conflict(objA, objB, ))
        self.assertIsNone(self.rules.eval_conflict(objA, objA2))


class TestFightSanity(unittest.TestCase):
    """Positive control: người giật cục + chồng lấn -> candidate (đúng hướng)."""

    def setUp(self):
        self.rules = PersonActionRules()

    def _jitter_boxes(self, base, n, amp, phase=0.0):
        boxes = []
        for i in range(n):
            off = amp * math.sin(i * 1.2 + phase)
            boxes.append([base[0] + off, base[1], base[2] + off, base[3]])
        return boxes

    def test_aggressive_motion_emits_candidate(self):
        boxA = self._jitter_boxes([180, 100, 300, 300], 12, amp=40)
        boxB = self._jitter_boxes([240, 100, 360, 300], 12, amp=40, phase=1.7)
        objA = make_obj(1, boxA)
        objB = make_obj(2, boxB)
        cand = None
        for _ in range(6):
            cand = self.rules.eval_conflict(objA, objB)
            if cand is not None:
                break
        self.assertIsNotNone(cand, "Va đập mạnh phải sinh candidate")
        self.assertEqual("HUMAN_CONFLICT", cand["event_type"])
        self.assertGreaterEqual(cand["confidence"],
                                config.FIGHT_CANDIDATE_THRESH)


if __name__ == "__main__":
    unittest.main()