"""Build a competition submission from the untouched official test template."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.io import atomic_write_json, load_json, require_distinct_paths
from aicomp_grounding.test_data import (
    OFFICIAL_QUERY_COUNT,
    OFFICIAL_TEMPLATE_CANONICAL_SHA256,
    validate_official_test_template,
)

DEFAULT_TEMPLATE = Path("data/Test/queries/queries.json")


def load_predictions(path: Path) -> dict[str, list[float] | None]:
    return load_json(path)


def _write_verified_zip(result_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{zip_path.name}.",
            suffix=".tmp",
            dir=zip_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(result_path, arcname="result.json")
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.namelist() != ["result.json"]:
                raise RuntimeError("Submission ZIP has an unexpected file layout")
            archived_raw = json.loads(archive.read("result.json").decode("utf-8"))
        expected = load_json(result_path)
        if archived_raw != expected:
            raise RuntimeError("Submission ZIP verification failed")
        os.replace(temporary, zip_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_submission(
    test_json_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    default_bbox: list[float] | None = None,
    allow_fallback: bool = False,
    expected_query_count: int | None = OFFICIAL_QUERY_COUNT,
    expected_template_sha256: str | None = OFFICIAL_TEMPLATE_CANONICAL_SHA256,
) -> Path:
    """Preserve every official field, add only bbox, and create a submission ZIP."""
    result_name = "result.diagnostic.json" if allow_fallback else "result.json"
    zip_name = "submission.diagnostic.zip" if allow_fallback else "submission.zip"
    result_path = output_dir / result_name
    zip_path = output_dir / zip_name
    for source in (test_json_path, predictions_path):
        require_distinct_paths(source, result_path)
        require_distinct_paths(source, zip_path)
    require_distinct_paths(result_path, zip_path)

    template = load_json(test_json_path)
    validate_official_test_template(
        template,
        expected_query_count=expected_query_count,
        expected_template_sha256=expected_template_sha256,
    )
    predictions = load_predictions(predictions_path)
    fallback = (
        validate_bbox(default_bbox)
        if default_bbox is not None
        else [0.0, 0.0, 0.001, 0.001]
    )
    if fallback is None:
        raise ValueError(f"Invalid fallback bbox: {default_bbox!r}")

    template_ids = set(template)
    prediction_ids = set(predictions)
    missing = template_ids - prediction_ids
    extra = prediction_ids - template_ids
    invalid = sorted(
        query_id
        for query_id in template_ids & prediction_ids
        if validate_bbox(predictions[query_id]) is None
    )
    if (missing or extra or invalid) and not allow_fallback:
        raise ValueError(
            "Final submission requires exact IDs and valid bboxes: "
            f"missing={len(missing)}, extra={len(extra)}, invalid={len(invalid)}. "
            "Use --allow-fallback only for a diagnostic submission."
        )

    submission = {}
    for query_id, original in template.items():
        bbox = validate_bbox(predictions.get(query_id))
        entry = dict(original)
        entry["bbox"] = bbox if bbox is not None else fallback
        submission[query_id] = entry
        if {key: entry[key] for key in original} != original:
            raise AssertionError(f"Non-bbox field changed for {query_id!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_path, submission)
    _write_verified_zip(result_path, zip_path)
    result_path.unlink()
    valid_count = len(template) - len(missing) - len(invalid)
    print(f"[build_submission] {len(template)} queries, {valid_count} valid predictions")
    if missing or extra or invalid:
        print(
            f"[build_submission] Diagnostic fallbacks: missing={len(missing)}, "
            f"extra={len(extra)}, invalid={len(invalid)}"
        )
    print(f"[build_submission] ZIP: {zip_path}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submission"))
    parser.add_argument(
        "--default-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback boxes only for an explicitly incomplete diagnostic package.",
    )
    args = parser.parse_args()
    build_submission(
        test_json_path=args.test_json,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        default_bbox=args.default_bbox,
        allow_fallback=args.allow_fallback,
    )


if __name__ == "__main__":
    main()
