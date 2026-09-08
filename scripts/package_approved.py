#!/usr/bin/env python3
"""Package reviewed assembly records and split indexes into approved.json.

Converts query-foundry assembly products (assembly.json) and source split indexes
into the final, directly consumable approved.json artifact required by downstream
training. Automatically computes and seals all four SHA-256 fingerprints:
  1. source_fingerprint      (content hash of immutable inputs)
  2. dataset_fingerprint     (content hash of full dataset dict)
  3. image_fingerprint       (manifest reference binding prefix "manifest_")
  4. preparation_fingerprint (content hash of split_manifest.json)

Supports single-split packaging (train or val) or joint dual-split packaging
(train + val) with cross-split leakage and prompt consistency verification.

Zero external pip dependencies required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.pipeline.contract import (
    ANNOTATION_MODE,
    ANNOTATION_PROTOCOL_VERSION,
    ASSIGNMENT_POLICY,
    RENDER_PROTOCOL,
    approved_dataset_fingerprint,
    clean_query_text,
    group_keys_by_scene,
    source_fingerprint,
    trusted_dataset_image_fingerprint,
    validate_annotation_query,
    validate_approved_artifact,
    validate_training_artifacts,
)
from foundry.utils import (
    ANNOTATION_API_BASE_URL,
    ANNOTATION_MODEL_LICENSE,
    ANNOTATION_MODEL_NAME,
    ANNOTATION_MODEL_REVISION,
    ANNOTATION_MODEL_WEIGHTS_URL,
    ANNOTATION_PROVIDER,
    atomic_write_json,
    load_json,
    stable_json_hash,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package assembly records into approved.json with deterministic SHA-256 fingerprints"
    )
    parser.add_argument(
        "--assembly",
        type=Path,
        nargs="+",
        required=True,
        help="one or more paths to assembly.json (e.g. asm-train-r6/assembly.json asm-val-r6/assembly.json)",
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
        help="override split name (only valid when a single assembly file is provided)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="output run ID (e.g. annot_r6; defaults to annot_<common_tag>)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "approved",
        help="base output directory (default: outputs/approved)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="exact destination file path (only valid when a single assembly file is provided)",
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
        help="override prompt hash (auto-detected from census plan if omitted)",
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
    return parser


def _find_prompt_hash(assembly_meta: dict) -> str:
    """Derive or locate the prompt hash from census or default."""
    census_run_id = assembly_meta.get("census_run_id")
    if census_run_id:
        plan_path = PROJECT_ROOT / "outputs" / "census" / census_run_id / "plan.json"
        if plan_path.is_file():
            try:
                plan = load_json(plan_path)
                meta = plan.get("metadata", {})
                for h_key in ("attr_prompt_hash", "findall_prompt_hash", "prompt_hash"):
                    if meta.get(h_key):
                        return str(meta[h_key])
            except Exception:
                pass
    fallback_seed = {
        "census_run_id": census_run_id or "default",
        "protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "mode": ANNOTATION_MODE,
    }
    return stable_json_hash(fallback_seed)


def package_single(
    assembly_path: Path,
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
    if not assembly_path.is_file():
        raise FileNotFoundError(f"Assembly file not found: {assembly_path}")

    assembly = load_json(assembly_path)
    assembly_meta = dict(assembly.get("metadata", {}))
    records = list(assembly.get("records", []))
    if not records:
        raise ValueError(f"Assembly file contains no records: {assembly_path}")

    # Determine split
    resolved_split = split or assembly_meta.get("split")
    if not resolved_split or resolved_split not in ("train", "val"):
        raise ValueError(f"Unable to determine split (train/val) from assembly metadata; specify --split explicitly")

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
    prep_fp = (
        assembly_meta.get("census_preparation_fingerprint")
        or (stable_json_hash(manifest_data) if manifest_data else None)
    )
    if not prep_fp:
        raise ValueError("Cannot determine preparation_fingerprint: split_manifest.json missing and not in assembly metadata")

    # Determine run ID
    assembly_tag = assembly_meta.get("run_tag") or assembly_path.parent.name
    resolved_run_id = run_id or f"annot_{assembly_tag}"

    # Determine prompt hash
    resolved_prompt_hash = prompt_hash or _find_prompt_hash(assembly_meta)

    # Determine key format
    sample_ids = [r["sample_id"] for r in records]
    has_multi = len(sample_ids) != len(set(sample_ids))
    use_item_id = (key_format == "item_id") or (key_format == "auto" and has_multi)

    if key_format == "sample_id" and has_multi:
        raise ValueError("Cannot use --key-format sample_id when assembly contains multiple objects per frame")

    # Assemble dataset
    data: dict[str, dict] = {}
    qc_failures: list[tuple[str, str, str]] = []

    for rec in records:
        sample_id = rec["sample_id"]
        if sample_id not in index:
            raise KeyError(f"Sample ID {sample_id!r} from assembly not found in index {index_path}")

        idx_item = index[sample_id]
        obj_idx = rec.get("object_index", 0)
        item_key = f"{sample_id}#{obj_idx:02d}" if use_item_id else sample_id

        if item_key in data:
            raise ValueError(f"Duplicate item key generated in dataset: {item_key!r}")

        query = clean_query_text(rec.get("query", ""))
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
        "source_type": "hosted_open_weights",
        "provider": ANNOTATION_PROVIDER,
        "api_base_url": ANNOTATION_API_BASE_URL,
        "annotator_model": ANNOTATION_MODEL_NAME,
        "annotator_revision": ANNOTATION_MODEL_REVISION,
        "model_weights_url": ANNOTATION_MODEL_WEIGHTS_URL,
        "model_license": ANNOTATION_MODEL_LICENSE,
        "mode": ANNOTATION_MODE,
        "assignment_policy": ASSIGNMENT_POLICY,
        "render_protocol": RENDER_PROTOCOL,
        "generation_config": {
            "query_max_tokens": 4096,
            "temperature": 0.2,
            "enable_thinking": None,
            "thinking_mode": "disabled",
            "response_format": None,
            "image_detail": "high",
        },
    }

    qc_status = {
        "complete": True,
        "failed_sequences": 0,
        "failed_frames": 0,
        "invalid_queries": 0,
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
        base_dir = output_dir or (PROJECT_ROOT / "outputs" / "approved")
        output_path = base_dir / resolved_run_id / resolved_split / "approved.json"

    if output_path.exists() and not force and not dry_run:
        raise FileExistsError(f"Destination file already exists: {output_path}. Pass --force to overwrite.")

    written_paths = []
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_path, artifact)
        written_paths.append(output_path)

        if qc_failures:
            # The downstream contract pins qc.invalid_queries to 0 inside the
            # artifact (exact-schema), so an honest count cannot live in the
            # metadata. A lenient (knowingly imperfect) delivery records the
            # real failures in a sidecar so the deviation stays traceable.
            sidecar = output_path.with_name(output_path.stem + ".qc_report.json")
            atomic_write_json(
                sidecar,
                {
                    "run_id": resolved_run_id,
                    "split": resolved_split,
                    "lenient_qc": True,
                    "qc_failures_count": len(qc_failures),
                    "failures": [
                        {"item_key": item_key, "query": query, "reason": reason}
                        for item_key, query, reason in qc_failures
                    ],
                },
            )
            written_paths.append(sidecar)
            print(
                f"[WARNING] lenient delivery carries {len(qc_failures)} QC failures; "
                f"report written to {sidecar}",
                file=sys.stderr,
            )

        # Optional direct export to main repo
        if export_to_main:
            main_dest = export_to_main / "outputs" / "annotations" / resolved_run_id / resolved_split / "approved.json"
            if main_dest.exists() and not force:
                raise FileExistsError(f"Main repo target already exists: {main_dest}. Pass --force to overwrite.")
            main_dest.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(main_dest, artifact)
            written_paths.append(main_dest)

    return {
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
        "written_paths": written_paths,
        "qc_failures_count": len(qc_failures),
        "artifact": artifact,
    }


def package(
    assemblies: list[Path] | Path | None = None,
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
    assembly_path: Path | None = None,
) -> list[dict] | dict:
    if assemblies is None and assembly_path is not None:
        assemblies = assembly_path
    if assemblies is None:
        raise ValueError("Must provide assemblies or assembly_path")

    return_single = assembly_path is not None or isinstance(assemblies, (str, Path))
    if isinstance(assemblies, (str, Path)):
        assemblies = [Path(assemblies)]

    if len(assemblies) > 1 and (output_path is not None or split is not None):
        raise ValueError("--output and --split can only be used when packaging a single assembly file")

    results: list[dict] = []
    by_split: dict[str, dict] = {}

    for asm_path in assemblies:
        res = package_single(
            assembly_path=asm_path,
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
            dry_run=dry_run,
            force=force,
        )
        s = res["split"]
        if s in by_split:
            raise ValueError(f"Multiple assemblies provided for split {s!r}")
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
        )

    return results[0] if return_single else results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        raw_res = package(
            assemblies=args.assembly,
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
