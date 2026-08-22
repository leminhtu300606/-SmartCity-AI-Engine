"""Rule 3 regression test — cadence scheduler.

Chứng minh bug đã sửa:
- fire/accident timer KHÔNG bị đẩy đi mỗi detect tick (trước đây fire/accident
  không bao giờ đến lượt chạy).
- Miss cadence chỉ đếm 1 lần / cadence bị bỏ (trước đây 5ms loop đếm miss
  hàng chục lần -> metric drop rate sai).
"""
import unittest

import numpy as np

import config
from inference.scheduler import InferenceScheduler


class FakeContext:
    def __init__(self):
        self.has_new = True
        self.next_detect_at = 0.0
        self.next_fire_at = 0.0
        self.next_accident_at = 0.0
        self.misses = 0
        self.taken = 0
        self.cid = "cam-test"

    @property
    def has_new_frame(self):
        return self.has_new

    def take_latest(self):
        self.taken += 1
        return np.zeros((384, 640, 3), dtype=np.uint8), self.taken, 0.0

    def record_missed_cadence(self):
        self.misses += 1

    def mark_ai_processed(self, idx):
        pass


class FakePool:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args, **kwargs):
        self.jobs.append(args)

    def shutdown(self, wait=False):
        pass


class TestSchedulerCadence(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeContext()
        self.sched = InferenceScheduler({"cam-test": self.ctx})
        self.pool = FakePool()
        self.sched.pool = self.pool
        self.cid = "cam-test"

    def tearDown(self):
        self.sched.stop()

    def test_fire_tick_actually_fires(self):
        # t=0: mọi stage đến hạn -> submit job, fire timer = fire interval
        self.sched._maybe_schedule(self.cid, self.ctx, 0.0)
        self.assertEqual(1, len(self.pool.jobs))
        _, _, _, _, run_detect_first, run_fire_first, run_acc_first = self.pool.jobs[0]
        self.assertTrue(run_detect_first)
        self.assertTrue(run_fire_first)
        self.assertTrue(run_acc_first)
        self.assertAlmostEqual(self.ctx.next_detect_at, 1.0 / config.AI_DETECT_FPS)
        fire_interval = 1.0 / config.AI_FIRE_FPS
        self.assertAlmostEqual(self.ctx.next_fire_at, fire_interval)

        # hoàn thành job; tick detect tiếp (t <= fire interval):
        # fire timer KHÔNG được advance thêm (bug cũ dời fire mãi mãi)
        self.sched._inflight[self.cid] = False
        self.sched._maybe_schedule(self.cid, self.ctx, fire_interval / 2.0)
        self.assertEqual(2, len(self.pool.jobs))
        _, _, _, _, run_detect_mid, run_fire_mid, _ = self.pool.jobs[1]
        self.assertTrue(run_detect_mid)
        self.assertFalse(run_fire_mid, "Tick detect giữa không được fire")
        self.assertAlmostEqual(self.ctx.next_fire_at, fire_interval)

        # đến đúng fire_interval -> fire thực sự chạy
        self.sched._inflight[self.cid] = False
        self.sched._maybe_schedule(self.cid, self.ctx, fire_interval)
        self.assertEqual(3, len(self.pool.jobs))
        _, _, _, _, run_detect_due, run_fire_due, _ = self.pool.jobs[2]
        self.assertTrue(run_fire_due, "Đến tick fire phải chạy fire")

    def test_fire_only_tick_skips_detect(self):
        # Rule 3: tick fire đến hạn NHƯNG detect chưa đến hạn -> job KHÔNG
        # chạy lại YOLO detect (giữ detection <= 5 FPS, không ~6.5 FPS).
        # detect interval = 0.2s; fire interval = 1/1.5 ≈ 0.6667s.
        for t in (0.0, 0.21, 0.42, 0.63):
            self.sched._maybe_schedule(self.cid, self.ctx, t)
            self.sched._inflight[self.cid] = False
        # t=0.667: fire due (>=0.6667) nhưng detect chưa due (next≈0.83)
        self.sched._maybe_schedule(self.cid, self.ctx, 0.667)
        self.assertEqual(5, len(self.pool.jobs))
        _, _, _, _, run_detect_only_fire, run_fire_only_fire, _ = self.pool.jobs[4]
        self.assertFalse(run_detect_only_fire,
                         "Tick fire giữa 2 tick detect KHÔNG được chạy YOLO")
        self.assertTrue(run_fire_only_fire)

    def test_miss_cadence_counted_once(self):
        # job trước chưa xong, detect đến hạn -> miss đếm ĐÚNG 1 lần
        self.sched._inflight[self.cid] = True
        self.sched._maybe_schedule(self.cid, self.ctx, 0.0)
        self.sched._maybe_schedule(self.cid, self.ctx, 0.005)
        self.assertEqual(1, self.ctx.misses, "Miss phải đếm 1 lần/cadence")
        self.assertAlmostEqual(
            self.ctx.next_detect_at, 1.0 / config.AI_DETECT_FPS)
        # lần loop sau (trước khi hết interval) không đếm thêm
        self.sched._maybe_schedule(self.cid, self.ctx, 0.010)
        self.assertEqual(1, self.ctx.misses)

    def test_detect_back_to_back_when_slow(self):
        # nếu pipeline chậm, job mới được đẩy ngay sau khi xong (bám đuổi cadence)
        self.sched._maybe_schedule(self.cid, self.ctx, 0.0)
        self.sched._inflight[self.cid] = False
        self.sched._maybe_schedule(self.cid, self.ctx, 100.0)
        self.assertEqual(2, len(self.pool.jobs))


if __name__ == "__main__":
    unittest.main()