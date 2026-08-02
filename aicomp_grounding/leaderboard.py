"""Leaderboard Score Registry and provenance binding for official competition submissions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile

from aicomp_grounding.io import atomic_write_json, load_json

DEFAULT_REGISTRY_PATH = Path("outputs/leaderboard_registry.json")


def load_leaderboard_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict:
    if not path.is_file():
        return {"version": 1, "submissions": []}
    return load_json(path)


def save_leaderboard_registry(data: dict, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    atomic_write_json(path, data)


def register_score(
    *,
    official_score: float,
    tag: str = "",
    notes: str = "",
    submission_zip: Path | None = None,
    inference_run_id: str = "",
    adapter_run_id: str = "",
    annotation_run_id: str = "",
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict:
    registry = load_leaderboard_registry(registry_path)
    zip_sha256 = ""
    zip_name = ""

    if submission_zip is not None and submission_zip.is_file():
        zip_name = submission_zip.name
        import hashlib
        zip_sha256 = hashlib.sha256(submission_zip.read_bytes()).hexdigest()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "official_score": float(official_score),
        "tag": tag.strip(),
        "notes": notes.strip(),
        "zip_name": zip_name,
        "zip_sha256": zip_sha256,
        "inference_run_id": inference_run_id.strip(),
        "adapter_run_id": adapter_run_id.strip(),
        "annotation_run_id": annotation_run_id.strip(),
    }

    registry["submissions"].append(record)
    save_leaderboard_registry(registry, registry_path)
    return record


def format_leaderboard_table(registry_path: Path = DEFAULT_REGISTRY_PATH) -> str:
    registry = load_leaderboard_registry(registry_path)
    submissions = registry.get("submissions", [])
    if not submissions:
        return "No registered submission scores found."

    lines = [
        "| Official Score | Tag | Inference Run ID | Adapter Run ID | Annotation Run ID | Notes |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for sub in submissions:
        score = f"{sub['official_score']:.4f}"
        tag = sub.get("tag", "-") or "-"
        infer = sub.get("inference_run_id", "-") or "-"
        adapter = sub.get("adapter_run_id", "-") or "-"
        annot = sub.get("annotation_run_id", "-") or "-"
        notes = sub.get("notes", "-") or "-"
        lines.append(f"| {score} | {tag} | `{infer}` | `{adapter}` | `{annot}` | {notes} |")
    return "\n".join(lines)
