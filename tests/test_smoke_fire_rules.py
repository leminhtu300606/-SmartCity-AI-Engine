"""Rule 6D/6E false-positive tests — cháy & khói.

KEY FALSE POSITIVES phải loại:
- "màu đỏ/cam/sáng" một mình KHÔNG phải FIRE (FireScore < 0.85).
- áo quần màu đỏ/cam (nằm trong bbox người) KHÔNG phải FIRE.
- "gray blur" tĩnh KHÔNG phải khói -> chỉ CANDIDATE, không bao giờ CONFIRMED.
Sanity: vùng khói đang LAN RỘNG phải confirm được.
"""
import unittest

import cv2
import numpy as np

import config
from events.confirm import EventConfirmTracker
from events.smoke_fire_rules import SmokeFireRules

_ROIS = [{
    "name": "ZoneFire",
    "event_type": config.EVENT_TYPE_SMOKE_FIRE,
    "polygon": [[0, 0], [640, 0], [640, 384], [0, 384]],
}]


def orange_frame():
    img = np.zeros((384, 640, 3), dtype=np.uint8)
    img[:] = (0, 128, 255)          # cam/đỏ rực, không nhấp nháy
    return img


def gray_rect(w, h):
    img = np.zeros((384, 640, 3), dtype=np.uint8)
    img[:h, :w] = (128, 128, 128)   # vùng xám tĩnh
    return img


class TestFireFalsePositive(unittest.TestCase):
    def setUp(self):
        self.rules = SmokeFireRules()

    def test_static_orange_never_confirmed_fire(self):
        # 20 frame cùng 1 vùng màu cam tĩnh -> không bao giờ FIRE_CONFIRMED.
        confirmer = EventConfirmTracker()
        wrong = False
        for _ in range(20):
            cand = self.rules.analyze_frame(orange_frame(), _ROIS,
                                            object_bboxes=[])
            if any(e.get("stage") == config.STAGE_CONFIRMED for e in
                   confirmer.process(cand, decay=True)):
                wrong = True
        self.assertFalse(wrong, "Màu cam tĩnh KHÔNG được CONFIRMED fire")

    def test_orange_clothing_inside_person_not_fire(self):
        # người mặc áo cam: pixel lửa nằm >60% trong bbox người -> loại (clothing)
        cand = self.rules.analyze_frame(
            orange_frame(), _ROIS,
            object_bboxes=[[0, 0, 640, 384]])
        self.assertEqual([], cand, "Áo quần đỏ/cam KHÔNG được là FIRE")


class TestSmokeFalsePositive(unittest.TestCase):
    def setUp(self):
        self.rules = SmokeFireRules()

    def test_static_gray_blob_never_confirmed_smoke(self):
        # "gray blur" tĩnh không lan -> SmokeScore < confirm -> không bao giờ CONFIRMED
        confirmer = EventConfirmTracker()
        wrong = False
        for _ in range(30):
            cand = self.rules.analyze_frame(gray_rect(400, 200), _ROIS,
                                            object_bboxes=[])
            for e in confirmer.process(cand, decay=True):
                # chỉ cần đảm bảo không có SMOKE CONFIRMED (cand khác có thể không)
                if e.get("event_type") == config.EVENT_TYPE_SMOKE and \
                        e.get("stage") == config.STAGE_CONFIRMED:
                    wrong = True
        self.assertFalse(wrong, "Khói tĩnh KHÔNG được CONFIRMED")

    def test_expanding_smoke_confirms(self):
        # Sanity: khói đang LAN RỘNG (expansion) -> SmokeScore cao -> CONFIRMED.
        rules = SmokeFireRules()
        confirmer = EventConfirmTracker()
        confirmed = False
        sizes = [(40, 30), (70, 45), (100, 60), (130, 75), (170, 95),
                 (210, 115), (260, 140), (310, 170), (370, 200), (430, 235),
                 (500, 270), (580, 315), (650, 360)]
        for w, h in sizes:
            cand = rules.analyze_frame(gray_rect(w, h), _ROIS,
                                       object_bboxes=[])
            for e in confirmer.process(cand, decay=True):
                if e.get("event_type") == config.EVENT_TYPE_SMOKE and \
                        e.get("stage") == config.STAGE_CONFIRMED:
                    confirmed = True
        self.assertTrue(confirmed, "Khói lan rộng phải CONFIRMED")


if __name__ == "__main__":
    unittest.main()