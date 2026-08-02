"""Deterministic artifact identity and strict metadata validation."""

from __future__ import annotations

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
    """Hash an unordered set of unique string keys."""
    values = list(keys)
    if not all(isinstance(key, str) for key in values):
        raise TypeError("Artifact keys must all be strings")
    if len(values) != len(set(values)):
        raise ValueError("Artifact keys must be unique")
    return stable_json_hash(sorted(values))


def file_set_fingerprint(data_root: str | Path, paths: Iterable[str | Path]) -> str:
    """Hash the relative names and complete bytes of a set of files under one root."""
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
    """Require every identity field to be present and equal."""
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
    """Require two metadata objects to have identical keys and values."""
    require_metadata_match(actual, expected, expected, label=label)
    extra = sorted(set(actual) - set(expected))
    if extra:
        raise ValueError(f"{label} metadata contains unexpected fields: {extra}")
