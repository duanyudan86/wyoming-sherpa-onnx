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


class TranscriptTests(unittest.TestCase):
    def test_qwen_metadata_stays_out_of_assist_commands(self):
        from app.asr_engine import Qwen3AsrEngine

        engine = object.__new__(Qwen3AsrEngine)
        engine.preserve_repetitions = True
        command = "Turn on the kitchen light."
        for raw in (command, "language English<asr_text>" + command, "Algerian language<asr_text>" + command):
            with self.subTest(raw=raw):
                self.assertEqual(engine._sanitize_text(raw), command)
        self.assertEqual(engine._sanitize_text("language None<ASR_TEXT>"), "")
