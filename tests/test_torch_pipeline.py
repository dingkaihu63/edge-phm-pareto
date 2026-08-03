from pathlib import Path
import sys
import unittest

import numpy as np
import torch


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

    def test_mean_pooling_uses_uniform_temporal_weights(self) -> None:
        model = build_model(6, 3, attention="mean", seed=11)
        x = torch.from_numpy(
            np.random.default_rng(11).normal(size=(4, 6, 3)).astype(np.float32)
        ).to(next(model.parameters()).device)
        with torch.no_grad():
            _, alpha = model.forward_attention(x)
        np.testing.assert_allclose(
            alpha.cpu().numpy(),
            np.full((4, 6, 1), 1.0 / 6.0),
            rtol=0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
