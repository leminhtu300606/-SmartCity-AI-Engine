"""Rule 3/4 cascade tests — classifier KHÔNG chạy object rules khi detect không chạy.

KEY:
- run_detect=False (tick fire/accident giữa 2 tick detect): fall/fight/intrusion
  KHÔNG được đánh giá trên dữ liệu cũ (tiết kiệm CPU) -> không bao giờ CONFIRMED.
- run_detect=True: object rules chạy -> ngã duy trì đủ frame -> CONFIRMED.
"""
import unittest

import config
from events.classifier import RuleBasedEventClassifier
from tracker.object_state import TrackedObjectState
from tracker.memory_manager import ObjectMemoryManager


def _falling_boxes(n=9):
    """Người NGÃ: đứng 4 frame rồi nằm ngang + dồn xuống dần (aspect cao)."""
    boxes = [[100, 100, 200, 300]] * 4
    for i in range(n - 4):
        y1 = 100 + i * 40
        boxes.append([100 - i * 40, y1, 320, y1 + 40])
    return boxes


def _make_memory(boxes, track_id=1):
    mm = ObjectMemoryManager(maxlen=config.TEMPORAL_BUFFER_MAXLEN,
                             ttl_frames=100)
    obj = TrackedObjectState(track_id, 0, maxlen=config.TEMPORAL_BUFFER_MAXLEN)
    obj.conf = 0.9
    t0 = 1000.0
    for i, b in enumerate(boxes):
        obj.update(list(b), t0 + i * 0.2)
    mm.objects[track_id] = obj
    return mm


class TestCascadeRunDetectGate(unittest.TestCase):
    def test_detect_off_never_confirms_fall(self):
        clf = RuleBasedEventClassifier(camera_id="cam09")
        mm = _make_memory(_falling_boxes())
        for frame in range(1, 20):
            confirmed = clf.evaluate(
                frame_bgr=None, memory_manager=mm, frame_idx=frame,
                run_detect=False, run_fire=False, run_vehicle=False)
            self.assertEqual(
                [], confirmed,
                "run_detect=False KHÔNG được CONFIRMED event object (Rule 3)")

    def test_detect_on_confirms_fall(self):
        clf = RuleBasedEventClassifier(camera_id="cam09")
        mm = _make_memory(_falling_boxes())
        # Ngã duy trì: mỗi frame đẩy thêm 1 box đang rơi xuống -> temporal tích luỹ
        confirmed_seen = False
        for frame in range(12):
            obj = mm.objects[1]
            obj.update(list([100 - 200 - frame * 40, 380 + frame * 10,
                             320, 420 + frame * 10]),
                       1000 + (frame + 20) * 0.2)
            confirmed = clf.evaluate(
                frame_bgr=None, memory_manager=mm, frame_idx=frame,
                run_detect=True, run_fire=False, run_vehicle=False)
            for ev in confirmed:
                if (ev.get("event_type") == "HUMAN_FALL"
                        and ev.get("stage") == config.STAGE_CONFIRMED):
                    confirmed_seen = True
        self.assertTrue(confirmed_seen,
                        "run_detect=True: ngã duy trì phải CONFIRMED")


if __name__ == "__main__":
    unittest.main()
