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


def load_json(path: Path | str) -> dict:
    path = Path(path)
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


def atomic_write_jsonl(path: Path, rows) -> None:
    """Durably replace ``path`` with one JSON object per line."""
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
            for row in rows:
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def load_jsonl(path: Path | str) -> list[dict]:
    """Read a JSONL file written by :func:`atomic_write_jsonl`."""
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {number} of {path}")
            rows.append(row)
    return rows


def require_distinct_paths(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError(f"Refusing to overwrite source JSON in place: {input_path}")
