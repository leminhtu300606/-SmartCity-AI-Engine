"""Rule 7/14: CANDIDATE != EVENT — test chi tiết EventConfirmTracker.

False-positive checks:
- score dưới ngưỡng candidate -> KHÔNG bao giờ confirm.
- chưa đủ số frame -> KHÔNG confirm (dù score cao).
- khi hết sign -> counter decay, không confirm vĩnh viễn.
"""
import unittest

import config
from events.confirm import EventConfirmTracker


def evt(event_type, score, track=None, zone=None):
    return {
        "event_type": event_type,
        "confidence": score,
        "track_ids": track or [],
        "zone_name": zone,
    }


class TestConfirmTracker(unittest.TestCase):
    def test_score_below_candidate_never_confirms(self):
        tracker = EventConfirmTracker()
        for _ in range(20):
            confirmed = tracker.process(
                [evt("HUMAN_FALL", 0.40, track=[1])], decay=True)
            self.assertEqual([], confirmed)
        self.assertEqual(tracker.pending_events, {})

    def test_single_high_score_frame_not_confirmed(self):
        tracker = EventConfirmTracker()
        c = tracker.process(
            [evt("HUMAN_FALL", 0.95, track=[1])], decay=True)
        self.assertEqual([], c)
        self.assertEqual(
            tracker.pending_events[(("HUMAN_FALL", (1,), None))]["count"], 1)

    def test_confirmed_only_after_enough_frames_and_score(self):
        tracker = EventConfirmTracker()
        meta = config.rule_meta("HUMAN_FALL")
        # đủ score nhưng chưa đủ frame
        for i in range(meta["frames"] - 1):
            confirmed = tracker.process(
                [evt("HUMAN_FALL", meta["confirm"], track=[1])], decay=True)
            self.assertEqual([], confirmed)
        # frame thứ `frames`: confirm
        confirmed = tracker.process(
            [evt("HUMAN_FALL", meta["confirm"], track=[1])], decay=True)
        self.assertEqual(1, len(confirmed))
        self.assertEqual(config.STAGE_CONFIRMED, confirmed[0]["stage"])

    def test_decay_after_event_disappears(self):
        tracker = EventConfirmTracker()
        for _ in range(config.rule_meta("HUMAN_CONFLICT")["frames"]):
            tracker.process(
                [evt("HUMAN_CONFLICT", 0.90, track=[1, 2])], decay=True)
        key = ("HUMAN_CONFLICT", (1, 2), None)
        self.assertIn(key, tracker.pending_events)
        # không còn xuất hiện -> decay dần rồi biến mất
        for _ in range(config.rule_meta("HUMAN_CONFLICT")["frames"] + 2):
            tracker.process([], decay=True)
        self.assertNotIn(key, tracker.pending_events)

    def test_confirm_requires_score_above_confirm_threshold(self):
        tracker = EventConfirmTracker()
        meta = config.rule_meta("FIRE_DETECTED")
        for _ in range(meta["frames"] + 2):
            confirmed = tracker.process(
                [evt("FIRE_DETECTED", meta["candidate"], zone="z")],
                decay=True)
            self.assertEqual([], confirmed)


if __name__ == "__main__":
    unittest.main()