from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_temporal_item_drift_v2 import select_counterparts, take_balanced  # noqa: E402


def row(item: str, direction: str, score: float) -> dict:
    return {
        "item_id": item,
        "drift_direction": direction,
        "drift_rank_score": score,
        "signed_popularity_log_ratio": score if direction == "gain" else -score,
    }


class DetectorV2Test(unittest.TestCase):
    def test_core_selection_balances_gain_and_loss(self) -> None:
        rows = [row(f"g{i}", "gain", 10 - i) for i in range(4)]
        rows += [row(f"l{i}", "loss", 10 - i) for i in range(4)]
        selected = take_balanced(rows, budget=6, gain_fraction=0.5)
        self.assertEqual(3, sum(value["drift_direction"] == "gain" for value in selected))
        self.assertEqual(3, sum(value["drift_direction"] == "loss" for value in selected))

    def test_core_selection_fills_missing_direction(self) -> None:
        rows = [row("g0", "gain", 5), row("l0", "loss", 4), row("l1", "loss", 3)]
        selected = take_balanced(rows, budget=3, gain_fraction=0.5)
        self.assertEqual({"g0", "l0", "l1"}, {value["item_id"] for value in selected})

    def test_counterpart_recovers_cross_window_neighborhood(self) -> None:
        core = [row("source", "loss", 3.0)]
        candidates = core + [row("target", "gain", 2.5), row("noise", "gain", 2.4)]
        left = defaultdict(Counter)
        right = defaultdict(Counter)
        left["source"] = Counter({"a": 4, "b": 2})
        right["target"] = Counter({"a": 8, "b": 4})
        right["noise"] = Counter({"z": 20})
        selected, info = select_counterparts(core, candidates, left, right, 1, 0.05)
        self.assertEqual(["target"], [value["item_id"] for value in selected])
        self.assertAlmostEqual(1.0, float(info["target"]["counterpart_similarity"]))


if __name__ == "__main__":
    unittest.main()
