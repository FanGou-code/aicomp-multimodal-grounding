#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicomp_grounding.contract import (
    ANNOTATION_PROTOCOL_VERSION,
    approved_dataset_fingerprint,
    source_fingerprint,
    validate_approved_artifact,
    validate_training_artifacts,
)
from aicomp_grounding.images import trusted_dataset_image_fingerprint
from aicomp_grounding.query import clean_query_text, validate_annotation_query
from aicomp_grounding.sharding import group_keys_by_scene
from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.io import atomic_write_json, load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package annotation datasets into approved.json with deterministic SHA-256 fingerprints"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=None,
        help="one or more paths to active annotation dataset JSON (e.g. outputs/annotations/train.json outputs/annotations/val.json)",
    )
    parser.add_argument(
        "--generation",
        type=Path,
        nargs="+",
        default=None,
        required=False,
        help="legacy: one or more paths to generation.json (e.g. asm-train-r6/generation.json asm-val-r6/generation.json)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="path to split index json (default: data/indexes/<split>.json)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="directory containing <split>.json files (default: data/indexes)",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "split_manifest.json",
        help="path to split_manifest.json (default: data/indexes/split_manifest.json)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default=None,
        help="override split name (only valid when a single generation file is provided)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="output run ID (e.g. annot_r6; defaults to annot_<common_tag>)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "annotations",
        help="base output directory (default: outputs/annotations)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="exact destination file path (only valid when a single generation file is provided)",
    )
    parser.add_argument(
        "--export-to-main",
        type=Path,
        default=None,
        help="optional path to main training repo root (e.g. ../aicomp-multimodal-grounding) to export directly",
    )
    parser.add_argument(
        "--prompt-hash",
        default=None,
        help="override prompt hash (default: hashed annotation protocol version)",
    )
    parser.add_argument(
        "--key-format",
        choices=("auto", "item_id", "sample_id"),
        default="auto",
        help="dataset key formatting policy (default: auto)",
    )
    parser.add_argument(
        "--lenient-qc",
        action="store_true",
        help="warn on query QC failures instead of raising an error",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print summary without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow overwriting existing output files",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="allow packaging datasets containing pending unannotated items",
    )
    return parser


def _find_prompt_hash() -> str:
    """Hash the annotation protocol as the train/val prompt-consistency key."""
    return stable_json_hash({"protocol_version": ANNOTATION_PROTOCOL_VERSION})


def package_single(
    generation_path: Path,
    index_path: Path | None = None,
    index_dir: Path | None = None,
    split_manifest_path: Path | None = None,
    split: str | None = None,
    run_id: str | None = None,
    output_path: Path | None = None,
    output_dir: Path | None = None,
    export_to_main: Path | None = None,
    prompt_hash: str | None = None,
    key_format: str = "auto",
    lenient_qc: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    if not generation_path.is_file():
        raise FileNotFoundError(f"Generation file not found: {generation_path}")

    generation = load_json(generation_path)
    generation_meta = dict(generation.get("metadata", {}))
    records = list(generation.get("records", []))
    if not records:
        raise ValueError(f"Generation file contains no records: {generation_path}")

    # Determine split
    resolved_split = split or generation_meta.get("split")
    if not resolved_split or resolved_split not in ("train", "val"):
        raise ValueError("Unable to determine split (train/val) from generation metadata; specify --split explicitly")

    # Locate index file
    if index_path is None:
        base_index_dir = index_dir or (PROJECT_ROOT / "data" / "indexes")
        index_path = base_index_dir / f"{resolved_split}.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Split index not found: {index_path}")
    index = load_json(index_path)

    # Locate split manifest
    manifest_data = None
    if split_manifest_path is None:
        base_index_dir = index_dir or (PROJECT_ROOT / "data" / "indexes")
        split_manifest_path = base_index_dir / "split_manifest.json"
    if split_manifest_path.is_file():
        manifest_data = load_json(split_manifest_path)

    # Determine preparation fingerprint
    prep_fp = stable_json_hash(manifest_data) if manifest_data else None
    if not prep_fp:
        raise ValueError("Cannot determine preparation_fingerprint: split_manifest.json missing and not in generation metadata")

    # Determine run ID
    generation_tag = generation_meta.get("run_tag") or generation_path.parent.name
    resolved_run_id = run_id or f"annot_{generation_tag}"

    # Determine prompt hash
    resolved_prompt_hash = prompt_hash or _find_prompt_hash()

    # Determine key format
    sample_ids = [r["sample_id"] for r in records]
    has_multi = len(sample_ids) != len(set(sample_ids))
    use_item_id = (key_format == "item_id") or (key_format == "auto" and has_multi)

    if key_format == "sample_id" and has_multi:
        raise ValueError("Cannot use --key-format sample_id when generation contains multiple objects per frame")

    # Build the dataset
    data: dict[str, dict] = {}
    qc_failures: list[tuple[str, str, str]] = []
    seen_queries: set[tuple[str, str]] = set()

    for rec in records:
        sample_id = rec["sample_id"]
        if sample_id not in index:
            raise KeyError(f"Sample ID {sample_id!r} from generation not found in index {index_path}")

        idx_item = index[sample_id]
        obj_idx = rec.get("object_index", 0)
        item_key = f"{sample_id}#{obj_idx:02d}" if use_item_id else sample_id

        if item_key in data:
            raise ValueError(f"Duplicate item key generated in dataset: {item_key!r}")

        query = clean_query_text(rec.get("query", ""))
        query_key = (idx_item["visible"], " ".join(query.split()).casefold())
        if rec.get("collision") or query_key in seen_queries:
            raise ValueError(f"Unresolved same-frame query collision at {item_key!r}")
        seen_queries.add(query_key)
        valid, reason = validate_annotation_query(query)
        if not valid:
            qc_failures.append((item_key, query, reason))

        # Build approved item dict
        bbox = rec.get("bbox")
        if bbox is None:
            bbox = idx_item.get("bbox")

        data[item_key] = {
            "visible": idx_item["visible"],
            "infrared": idx_item["infrared"],
            "depth": idx_item["depth"],
            "bbox": [float(v) for v in bbox],
            "width": int(idx_item["width"]),
            "height": int(idx_item["height"]),
            "query": query,
        }

    # Handle QC failures
    if qc_failures:
        print(f"\n[WARNING] Found {len(qc_failures)}/{len(data)} queries in {resolved_split} failing annotation QC rules:", file=sys.stderr)
        for ik, q, rsn in qc_failures[:5]:
            print(f"  - {ik}: {rsn} (query: {q!r})", file=sys.stderr)
        if len(qc_failures) > 5:
            print(f"  ... and {len(qc_failures) - 5} more", file=sys.stderr)

        if not lenient_qc:
            first_fail = qc_failures[0]
            raise ValueError(
                f"Query QC validation failed for {len(qc_failures)} samples in {resolved_split} (e.g. {first_fail[0]}: {first_fail[2]}). "
                f"Fix the queries or pass --lenient-qc to bypass query style gate."
            )

    # Compute fingerprints
    src_fp = source_fingerprint(data)
    data_fp = approved_dataset_fingerprint(data)
    img_fp = trusted_dataset_image_fingerprint(data, data, require_recorded_size=True)
    sequence_count = len(group_keys_by_scene(list(data), data))
    sample_count = len(data)

    provenance = {
        "source_type": "human_annotated",
    }

    qc_status = {
        "complete": True,
        "failed_sequences": 0,
        "failed_frames": 0,
        "invalid_queries": len(qc_failures),
        "generated_samples": sample_count,
    }

    artifact = {
        "metadata": {
            "status": "approved",
            "protocol_version": ANNOTATION_PROTOCOL_VERSION,
            "run_id": resolved_run_id,
            "split": resolved_split,
            "source_fingerprint": src_fp,
            "preparation_fingerprint": prep_fp,
            "image_fingerprint": img_fp,
            "dataset_fingerprint": data_fp,
            "sample_count": sample_count,
            "sequence_count": sequence_count,
            "prompt_hash": resolved_prompt_hash,
            "provenance": provenance,
            "qc": qc_status,
        },
        "data": data,
    }

    # Contract self-validation
    validate_approved_artifact(
        artifact,
        expected_split=resolved_split,
        expected_run_id=resolved_run_id,
        strict_query_qc=not lenient_qc,
    )

    # Determine destination paths
    if output_path is None:
        base_dir = output_dir or (PROJECT_ROOT / "outputs" / "annotations")
        output_path = base_dir / resolved_run_id / resolved_split / "approved.json"

    qc_report = None
    if qc_failures:
        qc_report = {
            "run_id": resolved_run_id, "split": resolved_split, "lenient_qc": True,
            "qc_failures_count": len(qc_failures),
            "failures": [{"item_key": k, "query": q, "reason": reason} for k, q, reason in qc_failures],
        }

    result = {
        "run_id": resolved_run_id,
        "split": resolved_split,
        "sample_count": sample_count,
        "sequence_count": sequence_count,
        "source_fingerprint": src_fp,
        "dataset_fingerprint": data_fp,
        "image_fingerprint": img_fp,
        "preparation_fingerprint": prep_fp,
        "prompt_hash": resolved_prompt_hash,
        "output_path": output_path,
        "written_paths": [],
        "qc_failures_count": len(qc_failures),
        "artifact": artifact,
        "qc_report": qc_report,
    }

    if not dry_run:
        _publish_results([result], export_to_main=export_to_main, force=force)
    return result


def _publish_results(results: list[dict], *, export_to_main: Path | None, force: bool) -> None:
    """Check every destination before writing any already-validated artifact."""
    deliveries: dict[Path, dict] = {}
    for result in results:
        destinations = [Path(result["output_path"])]
        if export_to_main is not None:
            destinations.append(export_to_main / "outputs" / "annotations" / result["run_id"] / result["split"] / "approved.json")
        for destination in destinations:
            payloads = [(destination, result["artifact"])]
            if result["qc_report"] is not None:
                payloads.append((destination.with_name(destination.stem + ".qc_report.json"), result["qc_report"]))
            for path, payload in payloads:
                path = path.resolve()
                if path in deliveries and deliveries[path] != payload:
                    raise ValueError(f"Conflicting packaging destinations: {path}")
                if path.exists() and not force:
                    raise FileExistsError(f"Destination file already exists: {path}. Pass --force to overwrite.")
                deliveries[path] = payload
                if path not in result["written_paths"]:
                    result["written_paths"].append(path)
    for path, payload in deliveries.items():
        atomic_write_json(path, payload)


def _derive_common_run_id(generations: list[Path]) -> str:
    """Derive a single shared run ID for multi-split packaging when omitted."""
    tags: list[str] = []
    for p in generations:
        try:
            meta = load_json(p).get("metadata", {})
            tag = meta.get("run_tag") or p.parent.name
        except (OSError, ValueError, KeyError, TypeError):
            tag = p.parent.name
        tags.append(str(tag))
    cleaned: list[str] = []
    for t in tags:
        c = re.sub(r"[-_](train|val)([-_]|$)", r"\2", t)
        cleaned.append(c)
    if len(set(cleaned)) == 1 and cleaned[0]:
        return f"annot_{cleaned[0]}"
    return f"annot_{tags[0]}"


def package_dataset_single(
    dataset_path: Path,
    split_manifest_path: Path | None = None,
    split: str | None = None,
    run_id: str | None = None,
    output_path: Path | None = None,
    output_dir: Path | None = None,
    export_to_main: Path | None = None,
    prompt_hash: str | None = None,
    lenient_qc: bool = False,
    allow_pending: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_json(dataset_path)
    meta = dict(dataset.get("metadata", {}))
    data = dict(dataset.get("data", {}))
    if not data:
        raise ValueError(f"Dataset contains no entries: {dataset_path}")

    resolved_split = split or meta.get("split") or ("val" if "val" in dataset_path.stem else "train")
    if resolved_split not in ("train", "val"):
        raise ValueError("Unable to determine split (train/val); specify --split explicitly")

    manifest_data = None
    if split_manifest_path is None:
        split_manifest_path = PROJECT_ROOT / "data" / "indexes" / "split_manifest.json"
    if split_manifest_path.is_file():
        manifest_data = load_json(split_manifest_path)
    prep_fp = stable_json_hash(manifest_data) if manifest_data else meta.get("preparation_fingerprint")

    resolved_run_id = run_id or meta.get("run_id") or f"annot_{dataset_path.stem}"
    resolved_prompt_hash = prompt_hash or meta.get("prompt_hash") or _find_prompt_hash()

    qc_failures: list[tuple[str, str, str]] = []
    seen_queries: set[tuple[str, str]] = set()

    for item_key, item in data.items():
        bbox = item.get("bbox")
        query = clean_query_text(item.get("query", ""))
        if not bbox or not query:
            qc_failures.append((item_key, query, "missing bbox or empty query"))
            continue
        query_key = (item["visible"], " ".join(query.split()).casefold())
        if query_key in seen_queries:
            raise ValueError(f"Unresolved same-frame query collision at {item_key!r}")
        seen_queries.add(query_key)
        valid, reason = validate_annotation_query(query)
        if not valid:
            qc_failures.append((item_key, query, reason))

    if qc_failures and not lenient_qc and not allow_pending:
        first_fail = qc_failures[0]
        raise ValueError(
            f"Query QC validation failed for {len(qc_failures)} samples in {resolved_split} (e.g. {first_fail[0]}: {first_fail[2]}). "
            f"Fix the queries or pass --lenient-qc / --allow-pending."
        )

    src_fp = source_fingerprint(data)
    data_fp = approved_dataset_fingerprint(data)
    img_fp = trusted_dataset_image_fingerprint(data, data, require_recorded_size=True)
    sequence_count = len(group_keys_by_scene(list(data), data))
    sample_count = len(data)

    metadata = {
        "status": "approved",
        "protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "run_id": resolved_run_id,
        "split": resolved_split,
        "source_fingerprint": src_fp,
        "image_fingerprint": img_fp,
        "dataset_fingerprint": data_fp,
        "sample_count": sample_count,
        "sequence_count": sequence_count,
        "prompt_hash": resolved_prompt_hash,
        "provenance": {"source_type": "human_annotated"},
        "qc": {
            "complete": True,
            "failed_sequences": 0,
            "failed_frames": 0,
            "invalid_queries": len(qc_failures),
            "generated_samples": sample_count,
        },
    }
    if prep_fp:
        metadata["preparation_fingerprint"] = prep_fp

    artifact = {
        "metadata": metadata,
        "data": data,
    }

    validate_approved_artifact(
        artifact,
        expected_split=resolved_split,
        expected_run_id=resolved_run_id,
        strict_query_qc=not lenient_qc,
        allow_pending=allow_pending,
    )

    base_out = output_dir or (PROJECT_ROOT / "outputs" / "annotations")
    resolved_output_path = output_path or (base_out / resolved_run_id / resolved_split / "approved.json")

    qc_report = None
    if qc_failures and lenient_qc:
        qc_report = {
            "run_id": resolved_run_id,
            "split": resolved_split,
            "lenient_qc": True,
            "qc_failures_count": len(qc_failures),
            "failures": [{"item_key": k, "query": q, "reason": reason} for k, q, reason in qc_failures],
        }

    result = {
        "run_id": resolved_run_id,
        "split": resolved_split,
        "sample_count": sample_count,
        "sequence_count": sequence_count,
        "source_fingerprint": src_fp,
        "dataset_fingerprint": data_fp,
        "image_fingerprint": img_fp,
        "preparation_fingerprint": prep_fp,
        "prompt_hash": resolved_prompt_hash,
        "output_path": resolved_output_path,
        "written_paths": [],
        "qc_failures_count": len(qc_failures),
        "artifact": artifact,
        "qc_report": qc_report,
    }

    if not dry_run:
        _publish_results([result], export_to_main=export_to_main, force=force)
    return result


def package(
    generations: list[Path] | Path | None = None,
    index_path: Path | None = None,
    index_dir: Path | None = None,
    split_manifest_path: Path | None = None,
    split: str | None = None,
    run_id: str | None = None,
    output_path: Path | None = None,
    output_dir: Path | None = None,
    export_to_main: Path | None = None,
    prompt_hash: str | None = None,
    key_format: str = "auto",
    lenient_qc: bool = False,
    allow_pending: bool = False,
    dry_run: bool = False,
    force: bool = False,
    generation_path: Path | None = None,
    datasets: list[Path] | Path | None = None,
    dataset_path: Path | None = None,
) -> list[dict] | dict:
    if datasets is None and dataset_path is not None:
        datasets = dataset_path
    if datasets is not None:
        return_single = dataset_path is not None or isinstance(datasets, (str, Path))
        if isinstance(datasets, (str, Path)):
            datasets = [Path(datasets)]

        if len(datasets) > 1 and (output_path is not None or split is not None):
            raise ValueError("--output and --split can only be used when packaging a single dataset file")

        if len(datasets) > 1 and run_id is None:
            run_id = _derive_common_run_id(datasets)

        results: list[dict] = []
        by_split: dict[str, dict] = {}
        for dpath in datasets:
            res = package_dataset_single(
                dataset_path=dpath,
                split_manifest_path=split_manifest_path,
                split=split,
                run_id=run_id,
                output_path=output_path,
                output_dir=output_dir,
                export_to_main=export_to_main,
                prompt_hash=prompt_hash,
                lenient_qc=lenient_qc,
                allow_pending=allow_pending,
                dry_run=True,
                force=force,
            )
            s = res["split"]
            if s in by_split:
                raise ValueError(f"Multiple datasets provided for split {s!r}")
            by_split[s] = res
            results.append(res)

        if "train" in by_split and "val" in by_split:
            target_run_id = results[0]["run_id"]
            for r in results:
                r["artifact"]["metadata"]["run_id"] = target_run_id
            validate_training_artifacts(
                by_split["train"]["artifact"],
                by_split["val"]["artifact"],
                annotation_run_id=target_run_id,
                strict_query_qc=not lenient_qc,
                allow_pending=allow_pending,
            )

        if not dry_run:
            _publish_results(results, export_to_main=export_to_main, force=force)
        return results[0] if return_single else results

    if generations is None and generation_path is not None:
        generations = generation_path
    if generations is None:
        raise ValueError("Must provide datasets, generations, dataset_path, or generation_path")

    return_single = generation_path is not None or isinstance(generations, (str, Path))
    if isinstance(generations, (str, Path)):
        generations = [Path(generations)]

    if len(generations) > 1 and (output_path is not None or split is not None):
        raise ValueError("--output and --split can only be used when packaging a single generation file")

    if len(generations) > 1 and run_id is None:
        run_id = _derive_common_run_id(generations)

    results: list[dict] = []
    by_split: dict[str, dict] = {}

    for asm_path in generations:
        res = package_single(
            generation_path=asm_path,
            index_path=index_path,
            index_dir=index_dir,
            split_manifest_path=split_manifest_path,
            split=split,
            run_id=run_id,
            output_path=output_path,
            output_dir=output_dir,
            export_to_main=export_to_main,
            prompt_hash=prompt_hash,
            key_format=key_format,
            lenient_qc=lenient_qc,
            dry_run=True,
            force=force,
        )
        s = res["split"]
        if s in by_split:
            raise ValueError(f"Multiple generations provided for split {s!r}")
        by_split[s] = res
        results.append(res)

    # Joint verification if both train and val are packaged together
    if "train" in by_split and "val" in by_split:
        target_run_id = results[0]["run_id"]
        for r in results:
            r["artifact"]["metadata"]["run_id"] = target_run_id
        validate_training_artifacts(
            by_split["train"]["artifact"],
            by_split["val"]["artifact"],
            annotation_run_id=target_run_id,
            strict_query_qc=not lenient_qc,
            allow_pending=allow_pending,
        )

    if not dry_run:
        _publish_results(results, export_to_main=export_to_main, force=force)
    return results[0] if return_single else results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    datasets = args.dataset
    generations = args.generation
    if datasets is None and generations is None:
        default_train = PROJECT_ROOT / "outputs" / "annotations" / "train.json"
        default_val = PROJECT_ROOT / "outputs" / "annotations" / "val.json"
        if default_train.is_file() and default_val.is_file():
            datasets = [default_train, default_val]
        else:
            parser.error("Must provide --dataset or --generation")

    try:
        raw_res = package(
            generations=generations,
            datasets=datasets,
            index_path=args.index,
            index_dir=args.index_dir,
            split_manifest_path=args.split_manifest,
            split=args.split,
            run_id=args.run_id,
            output_path=args.output,
            output_dir=args.output_dir,
            export_to_main=args.export_to_main,
            prompt_hash=args.prompt_hash,
            key_format=args.key_format,
            lenient_qc=args.lenient_qc,
            allow_pending=args.allow_pending,
            dry_run=args.dry_run,
            force=args.force,
        )
        results = [raw_res] if isinstance(raw_res, dict) else raw_res
    except Exception as exc:
        print(f"[ERROR] Packaging failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 64)
    print(f" Packaging Successful ({len(results)} split{'s' if len(results) > 1 else ''})")
    print("=" * 64)
    for result in results:
        print(f" Split [{result['split'].upper()}]: {result['run_id']}")
        print(f"  Samples:                 {result['sample_count']}")
        print(f"  Sequences:               {result['sequence_count']}")
        print(f"  source_fingerprint:      {result['source_fingerprint']}")
        print(f"  dataset_fingerprint:     {result['dataset_fingerprint']}")
        print(f"  image_fingerprint:       {result['image_fingerprint']}")
        print(f"  preparation_fingerprint: {result['preparation_fingerprint']}")
        print(f"  prompt_hash:             {result['prompt_hash']}")
        if args.dry_run:
            print("  [DRY RUN] No files written to disk.")
        else:
            for p in result["written_paths"]:
                print(f"  Wrote: {p}")
        print("-" * 64)
    print("=" * 64)


if __name__ == "__main__":
    main()
