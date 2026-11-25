"""Pinned English Fourier export for sherpa-onnx."""
import hashlib
from pathlib import Path

MODEL = "oruk-fourier-2b"
REPO = "oruk/oruk-fourier-2b"
REVISION = "3fda023e157ce7986f137f76fcf326ed6e08fa73"
SUBDIRECTORY = "onnx/sherpa-int8"
ROOT_CONFIG = "config.json"
ROOT_CONFIG_SIZE = 2479
ROOT_CONFIG_SHA256 = (
    "5bc1d99c141fc866b61efca97dcbc6412a38db25ff0934c38bf7b46fa09485a2"
)
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
    "tokenizer/vocab.json": 3080118,
}


def complete(directory: Path) -> bool:
    return all(
        (directory / name).is_file() and (directory / name).stat().st_size == size
        for name, size in SIZES.items()
    )


def has_root_config(directory: Path) -> bool:
    path = directory / ROOT_CONFIG
    if not path.is_file() or path.stat().st_size != ROOT_CONFIG_SIZE:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == ROOT_CONFIG_SHA256


def download(directory: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO,
        revision=REVISION,
        allow_patterns=[*[f"{SUBDIRECTORY}/{name}" for name in SIZES], ROOT_CONFIG],
        local_dir=directory,
    )
    result = directory / SUBDIRECTORY
    if not complete(result) or not has_root_config(directory):
        raise RuntimeError("Incomplete Fourier ONNX model or tokenizer")
    return result
