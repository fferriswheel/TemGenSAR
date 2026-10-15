from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_kuaisar_strict_temporal_dataset import window_bounds  # noqa: E402


class WindowBoundsTest(unittest.TestCase):
    def test_assigns_protocol_roles(self) -> None:
        windows = window_bounds(start_ms=1000, window_hours=1, count=6)
        self.assertEqual(
            ["initialization", "initialization", "initialization", "update", "update", "final_holdout"],
            windows["role"].tolist(),
        )
        self.assertEqual(1000 + 6 * 3_600_000, int(windows.iloc[-1]["end_exclusive_ms"]))


if __name__ == "__main__":
    unittest.main()
