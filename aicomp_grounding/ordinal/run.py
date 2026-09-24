"""Run identity, artifact paths, and the thinking sidecar.

Mirrors the inference run-identity pattern: one identity dict, hashed with the
fields that can change the artifact, prefixed per stage, and asserted against a
frozen field tuple.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from aicomp_grounding.artifacts import key_hash, stable_json_hash
from aicomp_grounding.io import atomic_write_jsonl, load_json, load_jsonl
from aicomp_grounding.ordinal import loader

ENUM_RUN_PREFIX = "ordinal_enum_"
RESOLVE_RUN_PREFIX = "ordinal_resolve_"

CHECKPOINT_VERSION = 1

#: Filename of the thinking sidecar, one JSON object per query.
THINKING_FILENAME = "thinking.jsonl"

#: Decoded visible images kept resident while walking items in path order.
SCENE_CACHE_MAX = 32

ENUM_METADATA_FIELDS = (
    "version",
    "run_id",
    "model",
    "model_revision",
    "instance_of",
    "max_new_tokens",
    "temperature",
    "think",
    "sampling",
    "prompts_fingerprint",
    "run_tag",
    "limit",
    "selected_key_hash",
    "input_fingerprint",
    "stats",
)

RESOLVE_METADATA_FIELDS = (
    "version",
    "run_id",
    "enum_run_id",
    "prediction_fingerprint",
    "axes",
    "iou_threshold",
    "run_tag",
    "selected_key_hash",
    "stats",
)


def build_enum_metadata(
    *,
    model: str,
    model_revision: str,
    max_new_tokens: int,
    temperature: float,
    think: bool,
    sampling: dict,
    run_tag: str,
    limit: int,
    selected_keys: Sequence[str],
    input_fingerprint: str,
    stats: dict | None = None,
) -> dict:
    """Identity of one enumeration run (base weights, RGB only).

    ``stats`` is written into the artifact but excluded from the run id.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    identity = {
        "version": CHECKPOINT_VERSION,
        "model": model,
        "model_revision": model_revision,
        "instance_of": None,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "think": bool(think),
        "sampling": dict(sampling),
        "prompts_fingerprint": loader.prompts_fingerprint(),
        "run_tag": run_tag,
        "limit": limit,
        "selected_key_hash": key_hash(selected_keys),
        "input_fingerprint": input_fingerprint,
    }
    metadata = {
        "version": CHECKPOINT_VERSION,
        "run_id": f"{ENUM_RUN_PREFIX}{stable_json_hash(identity, length=16)}",
        **identity,
        "stats": dict(stats or {}),
    }
    if tuple(metadata) != ENUM_METADATA_FIELDS:
        raise AssertionError("Ordinal enumeration metadata schema is out of sync")
    return metadata


def build_resolve_metadata(
    *,
    enum_run_id: str,
    prediction_fingerprint: str,
    iou_threshold: float,
    run_tag: str,
    selected_keys: Sequence[str],
    stats: dict | None = None,
) -> dict:
    """Identity of one resolve run: the single enumeration it consumed."""
    from aicomp_grounding.ordinal.resolve import AXES

    identity = {
        "version": CHECKPOINT_VERSION,
        "enum_run_id": enum_run_id,
        "prediction_fingerprint": prediction_fingerprint,
        "axes": list(AXES),
        "iou_threshold": iou_threshold,
        "run_tag": run_tag,
        "selected_key_hash": key_hash(selected_keys),
    }
    metadata = {
        "version": CHECKPOINT_VERSION,
        "run_id": f"{RESOLVE_RUN_PREFIX}{stable_json_hash(identity, length=16)}",
        **identity,
        "stats": dict(stats or {}),
    }
    if tuple(metadata) != RESOLVE_METADATA_FIELDS:
        raise AssertionError("Ordinal resolve metadata schema is out of sync")
    return metadata


def write_thinking(run_dir: Path, rows: dict[str, str]) -> None:
    """Write the thinking sidecar: one ``{"key", "thinking"}`` record per line."""
    atomic_write_jsonl(
        run_dir / THINKING_FILENAME,
        [{"key": key, "thinking": text} for key, text in rows.items()],
    )


def load_thinking(run_dir: Path) -> dict[str, str]:
    """Read the thinking sidecar, or an empty mapping when it is absent."""
    path = run_dir / THINKING_FILENAME
    if not path.is_file():
        return {}
    return {
        row["key"]: row["thinking"]
        for row in load_jsonl(path)
        if isinstance(row.get("key"), str) and isinstance(row.get("thinking"), str)
    }


def raw_depth_relpath(depth_rel: str) -> str:
    """Map a processed JET depth path back to the original 16-bit millimetre file.

    ``Processed/Test/depth_jet/x.png``          -> ``Test/Images/depth/x.png``
    ``Processed/Train/004/depth_jet/x.png``     -> ``Train/004/depth/x.png``
    """
    parts = Path(depth_rel).as_posix().split("/")
    if len(parts) < 4 or parts[0] != "Processed" or parts[-2] != "depth_jet":
        raise ValueError(f"Unrecognised processed depth path: {depth_rel!r}")
    middle = parts[1:-2]
    if middle == ["Test"]:
        return "/".join(["Test", "Images", "depth", parts[-1]])
    return "/".join([*middle, "depth", parts[-1]])


def load_axis_arrays(item: dict, data_dir: Path):
    """Return ``(depth_mm, infrared, image_size)`` for the pixel axes.

    Any missing file yields ``None`` for that array, which makes the matching
    axis unsupported and therefore leaves the box untouched.
    """
    import numpy
    from PIL import Image

    width, height = item.get("width"), item.get("height")
    size = (width, height) if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0 else None

    depth = None
    depth_rel = item.get("depth")
    if isinstance(depth_rel, str):
        try:
            path = data_dir / raw_depth_relpath(depth_rel)
        except ValueError:
            path = None
        if path is not None and path.is_file():
            with Image.open(path) as image:
                depth = numpy.asarray(image)

    infrared = None
    infrared_rel = item.get("infrared")
    if isinstance(infrared_rel, str):
        path = data_dir / infrared_rel
        if path.is_file():
            with Image.open(path) as image:
                infrared = numpy.asarray(image.convert("RGB"))

    if size is None:
        for array in (depth, infrared):
            if array is not None:
                size = (array.shape[1], array.shape[0])
                break
    return depth, infrared, size


def load_enum_artifacts(run_dir: Path) -> tuple[dict, dict, dict]:
    """Read one enumeration run as ``(metadata, parse, instances)``.

    The metadata must name the directory it was read from, so a run cannot be
    generated from mismatched artifacts.
    """
    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("run_id") != run_dir.name:
        raise ValueError(
            f"Enumeration run directory {run_dir.name!r} does not match its "
            f"metadata run_id {metadata.get('run_id')!r}"
        )
    return metadata, load_json(run_dir / "parse.json"), load_json(run_dir / "instances.json")
