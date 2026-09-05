"""Prepare an RGBDT dataset for visual grounding."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from pathlib import PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicomp_grounding.bbox import normalize_pixel_bbox
from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.config import PREPARATION_PROTOCOL_VERSION
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.io import load_json

from aicomp_grounding.test_data import (
    OFFICIAL_QUERY_COUNT,
    OFFICIAL_TEMPLATE_CANONICAL_SHA256,
    build_test_preparation_contract,
    validate_processed_test_index,
)


def parse_groundtruth(path: Path) -> dict[str, tuple[float, float, float, float]]:
    annotations = {}
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 5 or not parts[0]:
            raise ValueError(f"Malformed ground truth at {path}:{line_number}: {raw_line!r}")
        filename = parts[0]
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError(
                f"Ground-truth filename must be a basename at {path}:{line_number}: {filename!r}"
            )
        try:
            values = tuple(float(value) for value in parts[1:5])
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric ground truth at {path}:{line_number}: {raw_line!r}"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite ground truth at {path}:{line_number}: {raw_line!r}")
        if parts[0] in annotations:
            raise ValueError(f"Duplicate frame {parts[0]!r} in {path}")
        annotations[parts[0]] = values
    if not annotations:
        raise ValueError(f"Ground-truth file is empty: {path}")
    return annotations


def validate_preparation_config(
    *,
    min_depth_mm: int,
    max_depth_mm: int,
    scaling: str,
    train_ratio: float | None = None,
) -> None:
    if scaling not in {"fixed", "per-frame"}:
        raise ValueError(f"Unsupported depth scaling mode: {scaling!r}")
    if (
        isinstance(min_depth_mm, bool)
        or isinstance(max_depth_mm, bool)
        or not isinstance(min_depth_mm, int)
        or not isinstance(max_depth_mm, int)
        or not 0 <= min_depth_mm < max_depth_mm <= 65535
    ):
        raise ValueError("Depth limits must satisfy 0 <= min_depth_mm < max_depth_mm <= 65535")
    if train_ratio is not None and (
        isinstance(train_ratio, bool)
        or not isinstance(train_ratio, (int, float))
        or not math.isfinite(float(train_ratio))
        or not 0.0 < float(train_ratio) < 1.0
    ):
        raise ValueError("train_ratio must be a finite number between 0 and 1")


def validate_raw_depth(source_path: Path):
    import cv2
    import numpy as np

    depth = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Unreadable raw depth image: {source_path}")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(
            f"Raw depth must be single-channel uint16, got shape={depth.shape}, "
            f"dtype={depth.dtype}: {source_path}"
        )
    return depth


def validate_precolored_depth(source_path: Path):
    import cv2
    import numpy as np

    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unreadable pre-colored depth image: {source_path}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"Pre-colored depth must be uint8 HxWx3, got shape={image.shape}, "
            f"dtype={image.dtype}: {source_path}"
        )
    return image


def validate_processed_image(path: Path, *, expected_shape: tuple[int, int] | None = None):
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unreadable processed image: {path}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"Processed image must be uint8 HxWx3, got shape={image.shape}, "
            f"dtype={image.dtype}: {path}"
        )
    if expected_shape is not None and image.shape[:2] != expected_shape:
        raise ValueError(
            f"Processed image shape {image.shape[:2]} does not match {expected_shape}: {path}"
        )
    return image


def render_depth_to_jet(
    depth,
    *,
    min_depth_mm: int,
    max_depth_mm: int,
    scaling: str,
):
    import cv2
    import numpy as np

    validate_preparation_config(
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        scaling=scaling,
    )
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Depth array must be single-channel uint16")

    depth_float = depth.astype(np.float32)
    invalid = (depth_float < min_depth_mm) | (depth_float > max_depth_mm)
    valid = ~invalid
    depth_8bit = np.zeros(depth.shape, dtype=np.uint8)

    if valid.any():
        if scaling == "fixed":
            low, high = float(min_depth_mm), float(max_depth_mm)
        else:
            low, high = float(depth_float[valid].min()), float(depth_float[valid].max())
        if high > low:
            normalized = (depth_float - low) / (high - low) * 255.0
            depth_8bit = np.clip(normalized, 0, 255).astype(np.uint8)
            depth_8bit[invalid] = 0

    colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    colored[invalid] = (0, 0, 0)
    return colored


def validate_reusable_depth(
    raw_depth,
    output_path: Path,
    *,
    min_depth_mm: int,
    max_depth_mm: int,
    scaling: str,
) -> None:
    import numpy as np

    actual = validate_processed_image(output_path, expected_shape=raw_depth.shape)
    expected = render_depth_to_jet(
        raw_depth,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        scaling=scaling,
    )
    if not np.array_equal(actual, expected):
        raise ValueError(
            f"Existing processed depth does not match the requested conversion: {output_path}. "
            "Use --overwrite-depth to regenerate it."
        )


def copy_precolored_depth(source_path: Path, output_path: Path) -> bool:
    source = validate_precolored_depth(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with source_path.open("rb") as source_handle:
                while chunk := source_handle.read(1024 * 1024):
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        copied = validate_precolored_depth(temporary)
        if copied.shape != source.shape:
            raise ValueError(f"Copied pre-colored depth changed shape: {source_path}")
        os.replace(temporary, output_path)
        return True
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_reusable_precolored_depth(source_path: Path, output_path: Path) -> None:
    import numpy as np

    source = validate_precolored_depth(source_path)
    output = validate_precolored_depth(output_path)
    if not np.array_equal(source, output):
        raise ValueError(
            f"Existing processed JPG depth differs from its raw source: {output_path}. "
            "Use --overwrite-depth to restore it."
        )


def process_depth_to_jet(
    source_path: Path,
    output_path: Path,
    *,
    min_depth_mm: int,
    max_depth_mm: int,
    scaling: str,
) -> bool:
    import cv2

    depth = validate_raw_depth(source_path)
    colored = render_depth_to_jet(
        depth,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        scaling=scaling,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        if not cv2.imwrite(str(temporary), colored):
            return False
        validate_processed_image(temporary, expected_shape=depth.shape)
        os.replace(temporary, output_path)
        return True
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_split(sequence_names: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    validate_preparation_config(
        min_depth_mm=0,
        max_depth_mm=1,
        scaling="fixed",
        train_ratio=train_ratio,
    )
    if len(sequence_names) != len(set(sequence_names)):
        raise ValueError("Sequence names must be unique")
    if len(sequence_names) < 2:
        raise ValueError("At least two sequences are required for non-empty train and val splits")
    shuffled = list(sequence_names)
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * train_ratio)
    if not 0 < split_index < len(shuffled):
        raise ValueError("train_ratio produces an empty train or val split")
    return sorted(shuffled[:split_index]), sorted(shuffled[split_index:])


def _resolve_dataset_path(dataset_root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or ".." in posix_path.parts or "." in posix_path.parts:
        raise ValueError(f"{label} escapes or is not canonical: {relative!r}")
    root = dataset_root.resolve()
    resolved = (root / Path(*posix_path.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes dataset root: {relative!r}") from exc
    return resolved


def prepare_test_depth(args: argparse.Namespace) -> dict:
    validate_preparation_config(
        min_depth_mm=args.min_depth_mm,
        max_depth_mm=args.max_depth_mm,
        scaling=args.depth_scaling,
    )
    test_raw_root = args.dataset_root / "Test" / "Images" / "depth"
    test_processed_root = args.dataset_root / "Processed" / "Test" / "depth_jet"
    if not test_raw_root.is_dir():
        print(f"[prepare_rgbdt] No raw Test depth directory found at {test_raw_root}, skipping Test depth processing.")
        return {}

    depth_files = sorted(
        path
        for path in test_raw_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not depth_files:
        print(f"[prepare_rgbdt] No supported depth images found in {test_raw_root}.")
        return {}

    stats = {
        "total_png": sum(path.suffix.lower() == ".png" for path in depth_files),
        "total_precolored": sum(path.suffix.lower() != ".png" for path in depth_files),
        "written": 0,
        "would_write": 0,
        "reused": 0,
        "planned_depth_shapes": {},
    }
    for file_path in depth_files:
        output_path = test_processed_root / file_path.name
        is_raw_png = file_path.suffix.lower() == ".png"
        raw = (
            validate_raw_depth(file_path)
            if is_raw_png
            else validate_precolored_depth(file_path)
        )
        needs_write = not output_path.exists()
        if output_path.exists():
            try:
                if is_raw_png:
                    validate_reusable_depth(
                        raw,
                        output_path,
                        min_depth_mm=args.min_depth_mm,
                        max_depth_mm=args.max_depth_mm,
                        scaling=args.depth_scaling,
                    )
                else:
                    validate_reusable_precolored_depth(file_path, output_path)
            except ValueError:
                if not args.overwrite_depth:
                    raise
                needs_write = True
            else:
                stats["reused"] += 1

        if needs_write:
            if args.dry_run:
                relative_output = output_path.relative_to(args.dataset_root).as_posix()
                stats["planned_depth_shapes"][relative_output] = list(raw.shape[:2])
                stats["would_write"] += 1
            elif (
                process_depth_to_jet(
                    file_path,
                    output_path,
                    min_depth_mm=args.min_depth_mm,
                    max_depth_mm=args.max_depth_mm,
                    scaling=args.depth_scaling,
                )
                if is_raw_png
                else copy_precolored_depth(file_path, output_path)
            ):
                stats["written"] += 1
            else:
                raise RuntimeError(f"Failed to convert test depth image: {file_path}")

    return stats


def build_processed_test_index(official: dict) -> dict:
    """Map the official Test template to the in-memory processed worker layout."""
    processed = {}
    for query_id, item in official.items():
        depth_name = PurePosixPath(item["depth"]).name
        processed[query_id] = {
            "visible": f"Test/{item['visible']}",
            "infrared": f"Test/{item['infrared']}",
            "depth": f"Processed/Test/depth_jet/{depth_name}",
            "query": item["query"],
        }
    return processed


def validate_test_depth_references(
    dataset_root: Path,
    *,
    planned_depth_shapes: dict[str, list[int]] | None = None,
    expected_query_count: int | None = OFFICIAL_QUERY_COUNT,
    expected_template_sha256: str | None = OFFICIAL_TEMPLATE_CANONICAL_SHA256,
) -> list[str]:
    """Decode all indexed Test modalities and verify spatial alignment.

    PNG source depths are converted with JET. JPG sources are already uint8
    three-channel visualizations and are copied byte-for-byte. Returns a list
    of error strings (empty = all OK).
    """
    official_path = dataset_root / "Test" / "queries" / "queries.json"
    if not official_path.is_file():
        return [f"Official Test template not found at {official_path}"]
    try:
        official = load_json(official_path)
        data = build_processed_test_index(official)
        validate_processed_test_index(
            data,
            official,
            expected_query_count=expected_query_count,
            expected_template_sha256=expected_template_sha256,
        )
    except (ValueError, OSError) as exc:
        return [f"Processed test index does not match the official template: {exc}"]

    errors: list[str] = []
    invalid: list[str] = []
    import cv2

    decoded: dict[Path, tuple[int, int]] = {}
    for key, item in data.items():
        if not isinstance(key, str) or not isinstance(item, dict):
            invalid.append(f"{key!r}: entry is not an object")
            continue
        sizes = []
        relative_paths = []
        for field in ("visible", "infrared", "depth"):
            relative = item.get(field, "")
            if not isinstance(relative, str) or not relative:
                invalid.append(f"{key}: no {field} field")
                continue
            relative_paths.append(relative)
            try:
                path = _resolve_dataset_path(
                    dataset_root,
                    relative,
                    label=f"test entry {key!r} {field}",
                )
            except ValueError as exc:
                invalid.append(str(exc))
                continue
            if path not in decoded:
                planned_shape = (planned_depth_shapes or {}).get(relative)
                if field == "depth" and planned_shape is not None:
                    decoded[path] = tuple(planned_shape)
                else:
                    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        invalid.append(f"{key}: unreadable {field} {relative}")
                        continue
                    import numpy as np
                    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                        invalid.append(
                            f"{key}: {field} must be uint8 HxWx3, got "
                            f"shape={image.shape}, dtype={image.dtype}"
                        )
                        continue
                    decoded[path] = image.shape[:2]
            sizes.append(decoded[path])
        if len(relative_paths) == 3 and len(set(relative_paths)) != 3:
            invalid.append(f"{key}: modality paths must be distinct")
        if len(sizes) == 3 and len(set(sizes)) != 1:
            invalid.append(f"{key}: modality shapes differ {sizes}")

    if invalid:
        sample = "; ".join(invalid[:10])
        errors.append(
            f"{len(invalid)} test modality references missing, unreadable, or misaligned "
            f"(first 10: {sample}). Regenerate PNG depths with prepare_rgbdt.py; "
            "copy supplied JPG depth variants with prepare_rgbdt.py."
        )
    return errors


def prepare_dataset(
    args: argparse.Namespace,
    *,
    expected_test_query_count: int | None = OFFICIAL_QUERY_COUNT,
    expected_test_template_sha256: str | None = OFFICIAL_TEMPLATE_CANONICAL_SHA256,
) -> dict:
    validate_preparation_config(
        min_depth_mm=args.min_depth_mm,
        max_depth_mm=args.max_depth_mm,
        scaling=args.depth_scaling,
        train_ratio=args.train_ratio,
    )
    raw_root = args.dataset_root / "Train"
    processed_root = args.dataset_root / "Processed" / "Train"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw sequence directory: {raw_root}")

    destinations = [
        args.index_dir / "train.json",
        args.index_dir / "val.json",
        args.index_dir / "split_manifest.json",
    ]
    existing_indexes = [path for path in destinations if path.exists()]
    if not args.dry_run and existing_indexes and not args.overwrite_indexes:
        raise FileExistsError(
            f"Index output already exists: {existing_indexes[0]}. Use --overwrite-indexes explicitly."
        )

    import cv2

    sequences = sorted(path.name for path in raw_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not sequences:
        raise ValueError(f"No numeric sequence directories found under {raw_root}")
    train_sequences, val_sequences = build_split(sequences, args.train_ratio, args.seed)
    train_set = set(train_sequences)
    outputs = {"train": {}, "val": {}}
    stats = {
        "sequences": len(sequences),
        "groundtruth_rows": 0,
        "valid_samples": 0,
        "excluded_invalid_bbox": 0,
        "depth_written": 0,
        "depth_would_write": 0,
        "depth_reused": 0,
    }
    exclusions = []

    for sequence_index, sequence in enumerate(sequences, start=1):
        sequence_root = raw_root / sequence
        annotations = parse_groundtruth(sequence_root / "groundtruth.txt")
        stats["groundtruth_rows"] += len(annotations)
        target = outputs["train" if sequence in train_set else "val"]
        seen_stems: set[str] = set()

        for filename, (x, y, width, height) in sorted(annotations.items()):
            stem = Path(filename).stem
            if stem in seen_stems:
                raise ValueError(
                    f"Duplicate frame stem {stem!r} in sequence {sequence}; sample IDs would collide"
                )
            seen_stems.add(stem)
            visible_path = sequence_root / "color" / filename
            infrared_path = sequence_root / "infrared" / filename
            depth_path = sequence_root / "depth" / filename
            missing = [path for path in (visible_path, infrared_path, depth_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing modality for {sequence}/{filename}: {missing[0]}")

            visible = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
            infrared = cv2.imread(str(infrared_path), cv2.IMREAD_COLOR)
            depth = validate_raw_depth(depth_path)
            if visible is None or infrared is None:
                raise ValueError(f"Unreadable RGB/infrared image for {sequence}/{filename}")
            if not (visible.shape[:2] == infrared.shape[:2] == depth.shape):
                raise ValueError(
                    f"Misaligned modalities for {sequence}/{filename}: "
                    f"RGB={visible.shape[:2]}, IR={infrared.shape[:2]}, depth={depth.shape}"
                )
            image_height, image_width = visible.shape[:2]
            bbox = normalize_pixel_bbox(x, y, width, height, image_width, image_height)
            if bbox is None:
                stats["excluded_invalid_bbox"] += 1
                exclusions.append(
                    {
                        "sequence": sequence,
                        "filename": filename,
                        "bbox_xywh": [x, y, width, height],
                        "reason": "invalid_ground_truth_bbox",
                    }
                )
                continue

            depth_output = processed_root / sequence / "depth_jet" / filename
            needs_write = not depth_output.exists()
            if depth_output.exists():
                try:
                    validate_reusable_depth(
                        depth,
                        depth_output,
                        min_depth_mm=args.min_depth_mm,
                        max_depth_mm=args.max_depth_mm,
                        scaling=args.depth_scaling,
                    )
                except ValueError:
                    if not args.overwrite_depth:
                        raise
                    needs_write = True
                else:
                    stats["depth_reused"] += 1

            if needs_write:
                if args.dry_run:
                    stats["depth_would_write"] += 1
                elif process_depth_to_jet(
                        depth_path,
                        depth_output,
                        min_depth_mm=args.min_depth_mm,
                        max_depth_mm=args.max_depth_mm,
                        scaling=args.depth_scaling,
                    ):
                        stats["depth_written"] += 1
                else:
                    raise RuntimeError(f"Failed to convert train depth image: {depth_path}")

            sample_id = f"{sequence}_{stem}"
            if sample_id in target:
                raise ValueError(f"Duplicate output sample ID: {sample_id}")
            target[sample_id] = {
                "visible": f"Train/{sequence}/color/{filename}",
                "infrared": f"Train/{sequence}/infrared/{filename}",
                "depth": f"Processed/Train/{sequence}/depth_jet/{filename}",
                # Query is intentionally empty here: these indexes are the raw
                # annotation source list. Natural-language queries are produced
                # by the external private annotation pipeline and published into
                # outputs/annotations/<run_id>/<split>/approved.json. Training
                # only consumes approved.json, never these bare indexes.
                "query": "",
                "bbox": bbox,
                "width": image_width,
                "height": image_height,
            }
            stats["valid_samples"] += 1

        if sequence_index % 50 == 0 or sequence_index == len(sequences):
            print(f"Processed {sequence_index}/{len(sequences)} sequences")

    # Also prepare Test depth images
    test_stats = prepare_test_depth(args)
    stats["test_depth_written"] = test_stats.get("written", 0)
    stats["test_depth_would_write"] = test_stats.get("would_write", 0)
    stats["test_depth_reused"] = test_stats.get("reused", 0)
    if stats["valid_samples"] + stats["excluded_invalid_bbox"] != stats["groundtruth_rows"]:
        raise AssertionError("Prepared sample counts do not reconcile with ground-truth rows")
    # No test.json is generated; inference reads the official Test template
    # directly and maps its paths in memory.

    if not args.skip_test_validation:
        test_errors = validate_test_depth_references(
            args.dataset_root,
            planned_depth_shapes=test_stats.get("planned_depth_shapes", {}),
            expected_query_count=expected_test_query_count,
            expected_template_sha256=expected_test_template_sha256,
        )
        if test_errors:
            raise ValueError("Test dataset validation failed: " + "; ".join(test_errors))

    test_contract = None
    if not args.skip_test_validation and not args.dry_run:
        official_test = load_json(args.dataset_root / "Test" / "queries" / "queries.json")
        processed_test = build_processed_test_index(official_test)
        test_contract = build_test_preparation_contract(
            args.dataset_root,
            processed_test,
            official_test,
            expected_query_count=expected_test_query_count,
            expected_template_sha256=expected_test_template_sha256,
        )

    manifest = {
        "status": "complete",
        "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "depth_scaling": args.depth_scaling,
        "min_depth_mm": args.min_depth_mm,
        "max_depth_mm": args.max_depth_mm,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "stats": stats,
        "test_contract": test_contract,
        "exclusions": exclusions,
        "index_fingerprints": {
            "train": stable_json_hash(outputs["train"]),
            "val": stable_json_hash(outputs["val"]),
        },
        "index_sample_counts": {
            "train": len(outputs["train"]),
            "val": len(outputs["val"]),
        },
    }
    if not args.dry_run:
        args.index_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destinations[0], outputs["train"])
        atomic_write_json(destinations[1], outputs["val"])
        atomic_write_json(destinations[2], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare RGBDT tracking data without model/API calls")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing Train/")
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Output directory for train.json/val.json (default: <dataset-root>/indexes)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-depth-mm", type=int, default=300)
    parser.add_argument("--max-depth-mm", type=int, default=20000)
    parser.add_argument("--depth-scaling", choices=("fixed", "per-frame"), default="fixed")
    parser.add_argument("--overwrite-depth", action="store_true")
    parser.add_argument("--overwrite-indexes", action="store_true")
    parser.add_argument(
        "--skip-test-validation",
        action="store_true",
        help="Skip final Test index/image validation for an explicitly train-only dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.index_dir = (args.index_dir or args.dataset_root / "indexes").resolve()
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be between 0 and 1")
    if not 0 <= args.min_depth_mm < args.max_depth_mm <= 65535:
        parser.error("depth limits must satisfy 0 <= min < max <= 65535")
    manifest = prepare_dataset(args)
    print(manifest["stats"])
    if args.dry_run:
        print("Dry run complete; no depth images or indexes were written.")
    else:
        print(f"Indexes written to {args.index_dir}")


if __name__ == "__main__":
    main()
