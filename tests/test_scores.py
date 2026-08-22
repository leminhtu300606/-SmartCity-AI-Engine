"""Rule 6: weights từng công thức = đúng tỷ lệ (100%) và cấu hình hợp lệ.

M:  FallScore    = .25 posture + .25 vertical_motion + .20 aspect + .15 center_vel + .15 temporal
    FightScore   = .20 pair + .20 contact + .25 relative_motion + .20 intensity + .15 temporal
    Collision    = .15 stability + .20 rel_vel + .20 closing + .20 geometry + .15 vel_change + .10 temporal
    FireScore    = .35 model + .20 spatial + .20 temporal + .15 motion + .10 smoke
    SmokeScore   = .35 model + .25 temporal + .20 expansion + .20 shape
"""
import unittest

import config
from events import scores


class TestScoreWeights(unittest.TestCase):
    def _sum(self, keys):
        return sum(getattr(config, k) for k in keys)

    def test_fall_weights_sum_to_one(self):
        self.assertAlmostEqual(
            self._sum(["FALL_W_POSTURE", "FALL_W_VERTICAL_MOTION",
                       "FALL_W_ASPECT_CHANGE", "FALL_W_CENTER_VELOCITY",
                       "FALL_W_TEMPORAL"]), 1.0, places=6)

    def test_fight_weights_sum_to_one(self):
        self.assertAlmostEqual(
            self._sum(["FIGHT_W_PERSON_PAIR", "FIGHT_W_CONTACT",
                       "FIGHT_W_RELATIVE_MOTION", "FIGHT_W_MOTION_INTENSITY",
                       "FIGHT_W_TEMPORAL"]), 1.0, places=6)

    def test_collision_weights_sum_to_one(self):
        self.assertAlmostEqual(
            self._sum(["COLLISION_W_TRACK_STABILITY",
                       "COLLISION_W_RELATIVE_VELOCITY",
                       "COLLISION_W_DISTANCE_CLOSING",
                       "COLLISION_W_GEOMETRY",
                       "COLLISION_W_VELOCITY_CHANGE",
                       "COLLISION_W_TEMPORAL"]), 1.0, places=6)

    def test_fire_weights_sum_to_one(self):
        self.assertAlmostEqual(
            self._sum(["FIRE_W_MODEL", "FIRE_W_SPATIAL",
                       "FIRE_W_TEMPORAL", "FIRE_W_MOTION",
                       "FIRE_W_SMOKE"]), 1.0, places=6)

    def test_smoke_weights_sum_to_one(self):
        self.assertAlmostEqual(
            self._sum(["SMOKE_W_MODEL", "SMOKE_W_TEMPORAL",
                       "SMOKE_W_EXPANSION", "SMOKE_W_SHAPE"]), 1.0, places=6)

    def test_candidate_leq_confirm_for_all_events(self):
        for event_type, meta in config.EVENT_RULE_META.items():
            self.assertLessEqual(
                meta["candidate"], meta["confirm"],
                f"{event_type}: candidate phải <= confirm")
            self.assertGreaterEqual(meta["frames"], 1, event_type)

    def test_weighted_is_bounded(self):
        import numpy as np
        comps = {k: 0.7 for k in
                 ["posture", "vertical_motion", "bbox_aspect_change",
                  "center_velocity", "temporal"]}
        w = {k: 0.2 for k in comps}
        self.assertAlmostEqual(scores.weighted(comps, w), 0.7, places=6)


if __name__ == "__main__":
    unittest.main()