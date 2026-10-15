from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from export_kuaisar_initialization_jsonl import parse_token_ids, token_text  # noqa: E402


class TokenParsingTest(unittest.TestCase):
    def test_parses_anonymized_token_list(self) -> None:
        self.assertEqual([12, 34], parse_token_ids("[12, 34]"))
        self.assertEqual("tok12 tok34", token_text("[12, 34]"))

    def test_empty_tokens(self) -> None:
        self.assertEqual([], parse_token_ids("[]"))


if __name__ == "__main__":
    unittest.main()
