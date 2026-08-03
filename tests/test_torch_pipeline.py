from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from torch_common import build_model, mc_predict, predict_proba  # noqa: E402


class DropoutModeTest(unittest.TestCase):
    def test_deterministic_and_mc_dropout_modes(self) -> None:
        model = build_model(8, 4, dropout_rate=0.3, seed=7)
        x = np.random.default_rng(7).normal(size=(12, 8, 4)).astype(np.float32)

        deterministic_a = predict_proba(model, x)
        deterministic_b = predict_proba(model, x)
        mc_mean, mc_std = mc_predict(model, x, samples=8)
        deterministic_c = predict_proba(model, x)

        np.testing.assert_allclose(deterministic_a, deterministic_b, rtol=0, atol=0)
        np.testing.assert_allclose(deterministic_a, deterministic_c, rtol=0, atol=0)
        self.assertTrue(np.any(mc_std > 0))
        self.assertEqual(mc_mean.shape, deterministic_a.shape)


if __name__ == "__main__":
    unittest.main()
