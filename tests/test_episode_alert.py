from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from episode_alert_analysis import classify_episode, first_confirmed_alarm  # noqa: E402


class EpisodeAlertTest(unittest.TestCase):
    def test_confirmation_uses_confirmation_index(self) -> None:
        probabilities = np.array([0.1, 0.7, 0.8, 0.2])
        self.assertEqual(first_confirmed_alarm(probabilities, 0.5, confirm=2), 2)

    def test_premature_alert_is_not_on_time(self) -> None:
        outcome = classify_episode(
            probabilities=np.array([0.8, 0.9, 0.2, 0.7]),
            labels=np.array([0, 0, 0, 1]),
            threshold=0.5,
            confirm=1,
        )
        self.assertEqual(outcome["premature"], 1)
        self.assertEqual(outcome["on_time"], 0)
        self.assertEqual(outcome["missed"], 0)

    def test_positive_episode_can_be_on_time_or_missed(self) -> None:
        on_time = classify_episode(
            probabilities=np.array([0.1, 0.2, 0.8]),
            labels=np.array([0, 0, 1]),
            threshold=0.5,
            confirm=1,
        )
        missed = classify_episode(
            probabilities=np.array([0.1, 0.2, 0.3]),
            labels=np.array([0, 0, 1]),
            threshold=0.5,
            confirm=1,
        )
        self.assertEqual(on_time["on_time"], 1)
        self.assertEqual(on_time["delay"], 0.0)
        self.assertEqual(missed["missed"], 1)

    def test_false_alert_is_defined_only_for_negative_episode(self) -> None:
        false_alert = classify_episode(
            probabilities=np.array([0.1, 0.8]),
            labels=np.array([0, 0]),
            threshold=0.5,
            confirm=1,
        )
        quiet = classify_episode(
            probabilities=np.array([0.1, 0.2]),
            labels=np.array([0, 0]),
            threshold=0.5,
            confirm=1,
        )
        self.assertEqual(false_alert["false_alert"], 1)
        self.assertEqual(quiet["false_alert"], 0)
        self.assertFalse(false_alert["positive_episode"])


if __name__ == "__main__":
    unittest.main()
