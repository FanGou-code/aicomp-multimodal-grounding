"""Small, dependency-free helpers for durable JSON artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=_object_without_duplicate_keys)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def atomic_write_json(path: Path, data: dict, *, indent: int = 2) -> None:
    """Durably replace ``path`` without exposing a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def require_distinct_paths(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError(f"Refusing to overwrite source JSON in place: {input_path}")




import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def stable_json_hash(value: object, *, length: int | None = None) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest if length is None else digest[:length]


def key_hash(keys: Iterable[str]) -> str:
    
    values = list(keys)
    if not all(isinstance(key, str) for key in values):
        raise TypeError("Artifact keys must all be strings")
    if len(values) != len(set(values)):
        raise ValueError("Artifact keys must be unique")
    return stable_json_hash(sorted(values))


def file_set_fingerprint(data_root: str | Path, paths: Iterable[str | Path]) -> str:
    
    root = Path(data_root).resolve()
    resolved: dict[str, Path] = {}
    for raw_path in paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Artifact file escapes data root {root}: {candidate}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Artifact file not found: {candidate}")
        resolved[relative] = candidate

    digest = hashlib.sha256()
    for relative, path in sorted(resolved.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def require_metadata_match(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    required_fields: Iterable[str],
    *,
    label: str = "artifact",
) -> None:
    
    if not isinstance(actual, Mapping):
        raise ValueError(f"{label} metadata must be an object")

    fields = tuple(required_fields)
    missing_expected = [field for field in fields if field not in expected]
    if missing_expected:
        raise ValueError(f"Expected metadata is missing required fields: {missing_expected}")

    missing_actual = [field for field in fields if field not in actual]
    if missing_actual:
        raise ValueError(f"{label} metadata is missing required fields: {missing_actual}")

    mismatches = [
        field for field in fields
        if actual[field] != expected[field]
    ]
    if mismatches:
        details = ", ".join(
            f"{field}={actual[field]!r} (expected {expected[field]!r})"
            for field in mismatches
        )
        raise ValueError(f"{label} metadata mismatch: {details}")


def require_exact_metadata(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str = "artifact",
) -> None:
    
    require_metadata_match(actual, expected, expected, label=label)
    extra = sorted(set(actual) - set(expected))
    if extra:
        raise ValueError(f"{label} metadata contains unexpected fields: {extra}")




import platform
from importlib import metadata as importlib_metadata

ANNOTATION_PROVIDER = "zhipu"
ANNOTATION_MODEL_NAME = "glm-4.6v"
ANNOTATION_MODEL_REVISION = "2025-12-08"
ANNOTATION_MODEL_WEIGHTS_URL = "https://huggingface.co/zai-org/GLM-4.6V"
ANNOTATION_MODEL_LICENSE = "MIT"
ANNOTATION_API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ANNOTATION_MAX_TOKENS = 256
ANNOTATION_TEMPERATURE = 0.2
# Zhipu primarily enforces per-account concurrency, not RPM/TPM. These defaults
# keep the client-side limiter from throttling below concurrency; tune via CLI
# (--requests-per-minute / --tokens-per-minute) against your account's limits
# shown at https://bigmodel.cn/usercenter/proj-mgmt/rate-limits.
ANNOTATION_REQUESTS_PER_MINUTE = 600
ANNOTATION_TOKENS_PER_MINUTE = 500_000
ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST = 1_800

DATA_ROOT = "/data/data"
INFERENCE_COMPUTE_DTYPE = "bfloat16"
RUNTIME_PYTHON_VERSION = platform.python_version()

# Generic inference pixel-budget defaults recorded in run metadata (Qwen
# processor units, 28x28 per patch). Model adapters may override via kwargs.
INFERENCE_DEFAULT_MIN_PIXELS = 256 * 28 * 28
INFERENCE_DEFAULT_MAX_PIXELS = 3072 * 28 * 28

# Local development/validation dependencies are pinned in requirements-lock.txt.
# MODAL_GPU_PACKAGES is the separate Modal GPU runtime package set; pins are
# fallbacks only — current_runtime_packages() records actually-installed versions.
MODAL_GPU_PACKAGES = (
    "transformers==5.14.1",
    "accelerate==1.14.0",
    "peft==0.19.1",
    "qwen-vl-utils==0.0.14",
    "pillow==12.1.0",
    "torch==2.13.0",
    "torchvision==0.28.0",
)


def current_runtime_packages() -> list[str]:
    
    packages = list(MODAL_GPU_PACKAGES)

    for index, package in enumerate(packages):
        distribution, separator, _ = package.partition("==")
        if not separator:
            continue
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
        packages[index] = f"{distribution}=={version}"

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        packages = [
            f"torch=={torch.__version__}" if package.startswith("torch==") else package
            for package in packages
        ]
        try:
            hip_version = torch.version.hip
        except AttributeError:
            hip_version = None
        if hip_version:
            packages.append(f"hip=={hip_version}")

    return packages

CHECKPOINT_VERSION = 5
ANNOTATION_PROTOCOL_VERSION = 12
PREPARATION_PROTOCOL_VERSION = 2
TRAINING_PROTOCOL_VERSION = 2
MAX_MODAL_CONTAINERS = 10

INFERENCE_SPLITS = frozenset({"train", "val", "test"})
ANNOTATION_SPLITS = frozenset({"train", "val"})
