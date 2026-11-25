import hashlib
import sys
import tempfile
import types
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

    def test_download_includes_and_verifies_root_config(self):
        fake_sizes = {"decoder.onnx": 3, "tokenizer/config.json": 2}
        fake_config = b'{"model_type":"qwen3_asr"}'
        fake_sha = hashlib.sha256(fake_config).hexdigest()
        captured = {}

        def fake_snapshot_download(**kwargs):
            captured.update(kwargs)
            target = Path(kwargs["local_dir"])
            sub = target / fourier.SUBDIRECTORY
            (sub / "tokenizer").mkdir(parents=True, exist_ok=True)
            (sub / "decoder.onnx").write_bytes(b"abc")
            (sub / "tokenizer/config.json").write_bytes(b"{}")
            if not captured.get("omit_root_config"):
                (target / fourier.ROOT_CONFIG).write_bytes(fake_config)

        fake_hf = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"huggingface_hub": fake_hf}
        ), patch.object(fourier, "SIZES", fake_sizes), patch.object(
            fourier, "ROOT_CONFIG_SIZE", len(fake_config)
        ), patch.object(
            fourier, "ROOT_CONFIG_SHA256", fake_sha
        ):
            root = Path(tmp)
            result = fourier.download(root)
            self.assertEqual(result, root / fourier.SUBDIRECTORY)
            self.assertEqual(captured["repo_id"], fourier.REPO)
            self.assertEqual(captured["revision"], fourier.REVISION)
            self.assertIn(fourier.ROOT_CONFIG, captured["allow_patterns"])
            self.assertIn(
                f"{fourier.SUBDIRECTORY}/decoder.onnx", captured["allow_patterns"]
            )
            self.assertIn(
                f"{fourier.SUBDIRECTORY}/tokenizer/config.json",
                captured["allow_patterns"],
            )
            self.assertTrue(fourier.has_root_config(root))

            # Missing or corrupted root config must fail download verification.
            (root / fourier.ROOT_CONFIG).unlink()
            captured["omit_root_config"] = True
            with self.assertRaises(RuntimeError):
                fourier.download(root)

            (root / fourier.ROOT_CONFIG).write_bytes(b"x" * len(fake_config))
            with self.assertRaises(RuntimeError):
                fourier.download(root)

    def test_root_config_pin_matches_canonical_metadata(self):
        self.assertEqual(fourier.ROOT_CONFIG, "config.json")
        self.assertEqual(fourier.ROOT_CONFIG_SIZE, 2479)
        self.assertEqual(
            fourier.ROOT_CONFIG_SHA256,
            "5bc1d99c141fc866b61efca97dcbc6412a38db25ff0934c38bf7b46fa09485a2",
        )


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
