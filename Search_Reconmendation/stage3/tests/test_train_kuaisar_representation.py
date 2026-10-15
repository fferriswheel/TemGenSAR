from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_kuaisar_common_encoder import dropout_mask  # noqa: E402
from train_kuaisar_search_encoder import group_loss, parse_tokens  # noqa: E402


class KuaiSARRepresentationTest(unittest.TestCase):
    def test_token_parser(self) -> None:
        self.assertEqual(["tok12", "tok34"], parse_tokens("[12, 34]"))

    def test_dropout_keeps_one_real_token(self) -> None:
        mask = torch.tensor([[True, True], [False, False]])
        dropped = dropout_mask(mask, 1.0)
        self.assertTrue(dropped[0].any())
        self.assertFalse(dropped[1].any())

    def test_multi_positive_loss_rewards_either_positive(self) -> None:
        logits = torch.tensor([[5.0, 4.0, -2.0]])
        positives = torch.tensor([[0, 1]])
        positive_mask = torch.tensor([[True, True]])
        self.assertLess(float(group_loss(logits, positives, positive_mask)), 0.01)


if __name__ == "__main__":
    unittest.main()
