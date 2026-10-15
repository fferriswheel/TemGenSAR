import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from import_gensar_sid_mapping import parse_codes  # noqa: E402


def test_parse_codes_splits_shared_and_private_tokens():
    shared, private = parse_codes(
        ["<shared_a_1>", "<shared_b_2>", "<semantic_a_3>", "<semantic_b_4>"],
        "semantic",
    )
    assert shared == [1, 2]
    assert private == [3, 4]


def test_parse_codes_rejects_wrong_private_family():
    with pytest.raises(ValueError, match="Expected semantic"):
        parse_codes(["<shared_a_1>", "<collaborate_a_2>"], "semantic")
