"""Test: cấu hình dự án phải tuân thủ AI PERFORMANCE CONTRACT (Video2Action).

Khoá để dev không chỉnh config phá hợp đồng (vd AI_DETECT_FPS=25)
mà không bị bắt tại CI / startup.
"""
import config
import contract
import unittest


class ContractComplianceTest(unittest.TestCase):
    def test_config_satisfies_contract(self):
        violations = contract.validate(config)
        self.assertEqual(
            violations, [],
            "Cấu hình vi phạm hợp đồng: " + "; ".join(
                f"Rule {v.rule}: {v.message}" for v in violations))

    def test_hard_limits_are_below_caps(self):
        # Quy tắc 6/12/13: target phải <= hard limit
        self.assertLessEqual(
            config.PER_CAMERA_RAM_TARGET_MB, config.PER_CAMERA_RAM_HARD_MB)
        self.assertLessEqual(config.AI_DETECT_FPS, contract.DETECT_FPS_HARD + 1e-6)
        self.assertLessEqual(config.AI_MAX_THREADS, config.CPU_LOGICAL_THREADS)


if __name__ == "__main__":
    unittest.main()
