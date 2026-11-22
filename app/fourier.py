"""Pinned English Fourier export for sherpa-onnx."""
from pathlib import Path

MODEL = "oruk-fourier-2b"
REPO = "oruk/oruk-fourier-2b"
REVISION = "3fda023e157ce7986f137f76fcf326ed6e08fa73"
SUBDIRECTORY = "onnx/sherpa-int8"
SIZES = {
    "conv_frontend.onnx": 48080440,
    "decoder.int8.onnx": 5393503,
    "decoder.int8.onnx.data": 2032082944,
    "encoder.int8.onnx": 1670820,
    "encoder.int8.onnx.data": 306487296,
    "tokenizer/config.json": 2072,
    "tokenizer/merges.txt": 1671853,
    "tokenizer/tokenizer.json": 11429377,
    "tokenizer/tokenizer_config.json": 10386,
    "tokenizer/vocab.json": 3080118
}


def complete(directory: Path) -> bool:
    return all(
        (directory / name).is_file() and (directory / name).stat().st_size == size
        for name, size in SIZES.items()
    )


def download(directory: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO,
        revision=REVISION,
        allow_patterns=[f"{SUBDIRECTORY}/{name}" for name in SIZES],
        local_dir=directory,
    )
    result = directory / SUBDIRECTORY
    if not complete(result):
        raise RuntimeError("Incomplete Fourier ONNX model or tokenizer")
    return result
