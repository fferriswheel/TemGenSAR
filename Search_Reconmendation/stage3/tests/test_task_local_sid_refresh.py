from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from align_and_materialize_rec_suffix import (  # noqa: E402
    align_token_levels,
    selective_unique_assignment,
)


class TaskLocalSIDRefreshTest(unittest.TestCase):
    def test_hungarian_alignment_recovers_token_permutation(self) -> None:
        old = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
        new = np.asarray([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.int64)
        aligned, permutations, stats = align_token_levels(old, new, [2, 2], 0.0)
        np.testing.assert_array_equal(old, aligned)
        np.testing.assert_array_equal([1, 0], permutations[0])
        np.testing.assert_array_equal([1, 0], permutations[1])
        self.assertEqual(1.0, stats[0]["aligned_item_agreement"])

    def test_selective_assignment_preserves_unselected_and_uniqueness(self) -> None:
        prefix = np.asarray([[0], [0], [0], [1]], dtype=np.int64)
        old = np.asarray([[0, 0], [0, 1], [0, 2], [0, 0]], dtype=np.int64)
        proposals = np.asarray([[0, 1], [0, 1], [0, 0], [0, 1]], dtype=np.int64)
        selected = np.asarray([True, False, True, False])
        confidence = np.asarray([0.9, 0.0, 0.8, 0.0])
        result, stats = selective_unique_assignment(
            prefix, old, proposals, selected, confidence
        )
        np.testing.assert_array_equal(old[~selected], result[~selected])
        keys = [tuple(p) + tuple(s) for p, s in zip(prefix, result)]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(stats["unchanged_outside_selection"])


if __name__ == "__main__":
    unittest.main()
