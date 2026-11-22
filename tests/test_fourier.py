import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import fourier


class ModelDirectoryTests(unittest.TestCase):
    def test_external_data_is_required_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            fourier, "SIZES", {"decoder.onnx": 3, "decoder.onnx.data": 2}
        ):
            root = Path(tmp)
            (root / "decoder.onnx").write_bytes(b"abc")
            self.assertFalse(fourier.complete(root))
            (root / "decoder.onnx.data").write_bytes(b"x")
            self.assertFalse(fourier.complete(root))
            (root / "decoder.onnx.data").write_bytes(b"yz")
            self.assertTrue(fourier.complete(root))
