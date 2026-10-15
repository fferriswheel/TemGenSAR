from __future__ import annotations

import sys
import unittest
from pathlib import Path

import duckdb


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_kuaisar_temporal_pilot import (  # noqa: E402
    complete_window_count,
    create_balanced_train_catalog,
)


class CompleteWindowCountTest(unittest.TestCase):
    def test_keeps_nearly_complete_final_window(self) -> None:
        self.assertEqual(9, complete_window_count(215.99, 24))
        self.assertEqual(3, complete_window_count(215.99, 72))

    def test_drops_materially_partial_final_window(self) -> None:
        self.assertEqual(8, complete_window_count(8.5 * 24, 24))


class BalancedTrainCatalogTest(unittest.TestCase):
    def test_uses_only_initialization_support_and_unions_task_top_k(self) -> None:
        connection = duckdb.connect()
        connection.execute(
            "CREATE TABLE rec_positive(user_id INTEGER, item_id INTEGER, ts BIGINT)"
        )
        connection.execute(
            "CREATE TABLE search_positive(user_id INTEGER, item_id INTEGER, ts BIGINT)"
        )
        connection.execute(
            "INSERT INTO rec_positive VALUES (1, 10, 1), (2, 10, 2), "
            "(1, 20, 3), (1, 99, 1000)"
        )
        connection.execute(
            "INSERT INTO search_positive VALUES (1, 20, 1), (2, 20, 2), "
            "(1, 30, 3), (1, 98, 1000)"
        )
        create_balanced_train_catalog(
            connection,
            start_ms=0,
            window_hours=1,
            initialization_end_window=0,
            per_task_top_k=1,
        )
        rows = connection.execute(
            "SELECT item_id, selected_by FROM balanced_train_catalog ORDER BY item_id"
        ).fetchall()
        self.assertEqual([(10, "rec"), (20, "search")], rows)


if __name__ == "__main__":
    unittest.main()
