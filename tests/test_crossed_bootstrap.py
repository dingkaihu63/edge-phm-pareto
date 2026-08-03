from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from five_seed_bootstrap import MODELS, resample_seed  # noqa: E402


def unit_record(probability_shift: float):
    units = []
    for offset in (0.0, 0.1):
        probabilities = {
            model: np.array([0.1 + probability_shift + offset, 0.9])
            for model in MODELS
        }
        units.append({"y": np.array([0, 1]), "p": probabilities})
    return {
        "units": units,
        "thresholds": {model: 0.5 for model in MODELS},
    }


class CrossedBootstrapTest(unittest.TestCase):
    def test_common_unit_indices_can_be_reused_across_seed_records(self) -> None:
        indices = np.array([1, 1], dtype=int)
        first = resample_seed(unit_record(0.0), indices)
        second = resample_seed(unit_record(0.2), indices)
        self.assertEqual(set(first), set(MODELS))
        self.assertEqual(set(second), set(MODELS))
        self.assertTrue(all(value is not None for value in first.values()))
        self.assertTrue(all(value is not None for value in second.values()))


if __name__ == "__main__":
    unittest.main()
