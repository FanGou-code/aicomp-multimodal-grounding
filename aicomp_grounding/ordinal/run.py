"""Run identity, artifact paths, and the per-query group decision.

Mirrors the inference run-identity pattern: one identity dict, hashed with the
fields that can change the artifact, prefixed per stage, and asserted against a
frozen field tuple so a new field can never slip in unhashed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from aicomp_grounding.artifacts import key_hash, stable_json_hash
from aicomp_grounding.io import load_json
from aicomp_grounding.ordinal import loader

if TYPE_CHECKING:  # resolve imports this module, so the dependency stays one-way
    from aicomp_grounding.ordinal.resolve import Decision

ENUM_RUN_PREFIX = "ordinal_enum_"
RESOLVE_RUN_PREFIX = "ordinal_resolve_"

CHECKPOINT_VERSION = 1

#: How many serving models must independently replace the box before the round
#: is adopted.
MIN_REPLACES = 2

#: Independent enumeration runs per model.  Two is the minimum that can disagree.
ENUM_RUNS = 2

AXIS_SET_VERSION = 1

#: Decoded visible images kept resident while walking items in path order, so
#: one image is decoded once for both enumeration runs.
SCENE_CACHE_MAX = 32

ENUM_METADATA_FIELDS = (
    "version",
    "run_id",
    "model",
    "model_revision",
    "instance_of",
    "max_new_tokens",
    "temperature",
    "enum_runs",
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
    "enum_run_ids",
    "prediction_fingerprints",
    "axes",
    "iou_threshold",
    "min_replaces",
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
    run_tag: str,
    limit: int,
    selected_keys: Sequence[str],
    input_fingerprint: str,
    stats: dict | None = None,
) -> dict:
    """Identity of one model's enumeration run (base weights, RGB only).

    ``stats`` is written into the artifact but excluded from the run id, so a
    run keeps its identity while its counters are filled in after the loop.
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
        "enum_runs": ENUM_RUNS,
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
    enum_run_ids: Sequence[str],
    prediction_fingerprints: Sequence[str],
    iou_threshold: float,
    run_tag: str,
    selected_keys: Sequence[str],
    stats: dict | None = None,
) -> dict:
    """Identity of one resolve run: every upstream run it consumed."""
    if len(enum_run_ids) != len(prediction_fingerprints):
        raise ValueError("Every enumeration run needs its base prediction fingerprint")
    from aicomp_grounding.ordinal.resolve import AXES

    identity = {
        "version": CHECKPOINT_VERSION,
        "enum_run_ids": list(enum_run_ids),
        "prediction_fingerprints": list(prediction_fingerprints),
        "axes": list(AXES),
        "iou_threshold": iou_threshold,
        "min_replaces": MIN_REPLACES,
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


def adopt_replacements(decisions: Sequence[Decision]) -> bool:
    """True when enough models independently replaced the box to adopt the round."""
    return sum(1 for decision in decisions if decision.action == "replace") >= MIN_REPLACES


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
        path = data_dir / raw_depth_relpath(depth_rel)
        if path.is_file():
            with Image.open(path) as image:
                depth = numpy.asarray(image)

    infrared = None
    infrared_rel = item.get("infrared")
    if isinstance(infrared_rel, str):
        path = data_dir / infrared_rel
        if path.is_file():
            with Image.open(path) as image:
                infrared = numpy.asarray(image.convert("RGB"))

    if size is None and depth is not None:
        size = (depth.shape[1], depth.shape[0])
    return depth, infrared, size


def load_enum_artifacts(run_dir: Path) -> tuple[dict, dict, dict]:
    """Read one enumeration run as ``(metadata, parse, instances)``.

    The metadata must name the directory it was read from, so a run cannot be
    assembled from mismatched artifacts.
    """
    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("run_id") != run_dir.name:
        raise ValueError(
            f"Enumeration run directory {run_dir.name!r} does not match its "
            f"metadata run_id {metadata.get('run_id')!r}"
        )
    return metadata, load_json(run_dir / "parse.json"), load_json(run_dir / "instances.json")
