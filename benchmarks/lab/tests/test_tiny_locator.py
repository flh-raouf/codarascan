from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from barcode_benchmark.tiny_locator import TinyBarcodeLocator


class TinyLocatorTests(unittest.TestCase):
    def test_preserves_spatial_shape_and_outputs_two_classes(self) -> None:
        model = TinyBarcodeLocator(width=8).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 96, 128))
        self.assertEqual(tuple(output.shape), (1, 2, 96, 128))
