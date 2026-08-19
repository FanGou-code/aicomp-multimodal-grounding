"""Standalone single-GPU offline inference and evaluation entry.

Model-agnostic: ``--model`` selects a grounding adapter (qwen3vl /
internvl35 / groundingdino / mock). Handles both:
1. Competition Submission Generation (when running on unannotated test queries)
2. Validation Accuracy & IoU Score Evaluation (when running on annotated val/train queries)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
import time

import torch

# This entrypoint lives in offline/; make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.inference_core import evaluate_predictions, load_inference_items
from aicomp_grounding.inference_state import (
    build_run_metadata,
    fingerprint_inputs,
    fingerprint_lora,
)
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.models import available_models, get_adapter
from aicomp_grounding.models.base import ModelInput
from aicomp_grounding.models.qwen3vl import MAX_PIXELS
from aicomp_grounding.paths import ProjectPaths, resolve_from_root
from aicomp_grounding.submission import build_submission

CACHE_MAX_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline grounding inference or evaluation on a single GPU."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3vl",
        choices=available_models(),
        help="Grounding model adapter to run.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional local base-model directory override (offline machines); "
        "defaults to the adapter's canonical hub id + pinned revision.",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=Path("data/test.json"),
        help="Path to input dataset JSON index (e.g. data/test.json or data/val.json).",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=MAX_PIXELS,
        help="Qwen processor max_pixels override. Training uses 3072 patches "
        "(2408448); lower it (e.g. 1920*28*28=1505280) if the inference GPU "
        "is short on VRAM. Lowering hurts grounding accuracy.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to root data directory containing Test/ and Processed/.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Optional LoRA adapter directory; omit for zero-shot runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of queries predicted per adapter call.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Root directory for inference outputs; results go to <output-dir>/<run_id>/.",
    )
    parser.add_argument(
        "--annotation-run-id",
        type=str,
        default="",
        help="Annotation run id when --test-json is an approved artifact (train/val).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Arbitrary experiment tag folded into the run id, matching Modal behavior.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, limit execution to the first N queries.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing predictions.json in output-dir if present.",
    )
    parser.add_argument(
        "--batch-save",
        type=int,
        default=50,
        help="Save predictions.json checkpoint every N queries.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root; relative input and output paths resolve from here.",
    )
    return parser.parse_args()


def resolve_image_path(raw_path: str, data_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (data_dir / p).resolve()


def load_scene_images(item: dict, data_dir: Path):
    from PIL import Image

    images_dict = item.get("images") if isinstance(item.get("images"), dict) else item
    v_raw = images_dict.get("visible")
    i_raw = images_dict.get("infrared")
    d_raw = images_dict.get("depth")

    if not v_raw or not i_raw or not d_raw:
        raise ValueError(f"Sample item missing modality image paths: {item}")

    v_path = resolve_image_path(v_raw, data_dir)
    i_path = resolve_image_path(i_raw, data_dir)
    d_path = resolve_image_path(d_raw, data_dir)

    if not v_path.is_file():
        raise FileNotFoundError(f"Visible image not found: {v_path}")
    if not i_path.is_file():
        raise FileNotFoundError(f"Infrared image not found: {i_path}")
    if not d_path.is_file():
        raise FileNotFoundError(f"Depth image not found: {d_path}")

    visible_img = Image.open(v_path).convert("RGB")
    infrared_img = Image.open(i_path).convert("RGB")
    depth_img = Image.open(d_path).convert("RGB")
    return visible_img, infrared_img, depth_img


def main():
    args = parse_args()

    paths = ProjectPaths.from_root(args.project_root)
    args.data_dir = resolve_from_root(args.data_dir, paths.root)
    args.test_json = resolve_from_root(args.test_json, paths.root)
    args.output_dir = resolve_from_root(args.output_dir, paths.root)
    if args.lora_path is not None:
        args.lora_path = resolve_from_root(args.lora_path, paths.root)

    if not args.test_json.is_file():
        raise FileNotFoundError(f"Dataset JSON index not found: {args.test_json}")

    adapter_kwargs = {}
    if args.model == "qwen3vl":
        adapter_kwargs["max_pixels"] = args.max_pixels
    adapter = get_adapter(args.model, **adapter_kwargs)

    # Accepts an approved annotation artifact or a flat {query_id: item}
    # index; see the shared inference core for the exact contract.
    items, approved_metadata = load_inference_items(args.test_json, limit=args.limit)

    # ---- Build a Modal-compatible run identity ----------------------------
    selected_keys = [item["key"] for item in items]
    adapter_dir = Path(args.lora_path).resolve() if args.lora_path else None
    if adapter_dir is not None and not adapter_dir.exists():
        print(f"Warning: LoRA path {adapter_dir} does not exist; running zero-shot.")
        adapter_dir = None
    if adapter_dir is not None and not adapter.supports_lora:
        print(f"Warning: --model {adapter.name} does not use LoRA; ignoring adapter path.")
        adapter_dir = None
    adapter_fingerprint = fingerprint_lora(adapter_dir)
    dataset_for_fp = {item["key"]: item for item in items}
    input_fingerprint = fingerprint_inputs(dataset_for_fp, selected_keys)

    metadata = build_run_metadata(
        mode="base",
        split="test" if approved_metadata is None else approved_metadata["split"],
        annotation_run_id=args.annotation_run_id,
        model=adapter.model_name,
        model_revision=adapter.model_revision,
        lora_path=adapter_dir,
        adapter_fingerprint=adapter_fingerprint,
        prompt_hash=adapter.prompt_hash(),
        generation_config=adapter.generation_config,
        run_tag=args.run_tag,
        limit=args.limit if args.limit > 0 else None,
        selected_keys=selected_keys,
        input_fingerprint=input_fingerprint,
        image_fingerprint="local",
        num_shards=1,
    )
    run_id = metadata["run_id"]
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.json"
    checkpoint_path = run_dir / "checkpoint.json"
    metadata_path = run_dir / "metadata.json"
    print(f"Run id: {run_id}")
    print(f"Output dir: {run_dir}")

    predictions: dict[str, list[float] | None] = {}
    if args.resume and predictions_path.is_file():
        print(f"Resuming from existing predictions file: {predictions_path}")
        predictions = load_json(predictions_path)

    pending_items = [item for item in items if item["key"] not in predictions]
    print(f"Total queries in dataset: {len(items)} | Already finished: {len(predictions)} | Pending: {len(pending_items)}")

    if pending_items:
        pending_items.sort(key=lambda x: (x.get("visible", x["key"]), x["key"]))

        print(f"Loading model '{adapter.model_name}' (revision {adapter.model_revision})...")
        adapter.load(device="cuda", lora_path=adapter_dir, model_path=args.model_path)

        scene_cache: OrderedDict = OrderedDict()
        batch: list[ModelInput] = []

        def flush():
            if not batch:
                return
            results = adapter.predict(batch)
            for sample, result in zip(batch, results):
                predictions[sample.key] = result.bbox
            batch.clear()

        start_time = time.time()
        processed_count = len(predictions)

        print("Starting single-GPU inference...")

        for item in pending_items:
            scene_id = (
                item["visible"],
                item["infrared"],
                item["depth"],
            )

            if scene_id in scene_cache:
                images = scene_cache[scene_id]
                scene_cache.move_to_end(scene_id)
            else:
                images = load_scene_images(item, args.data_dir)
                scene_cache[scene_id] = images
                if len(scene_cache) > CACHE_MAX_SIZE:
                    scene_cache.popitem(last=False)

            batch.append(
                ModelInput(
                    visible=images[0],
                    infrared=images[1],
                    depth=images[2],
                    query=item["query"],
                    key=item["key"],
                )
            )
            if len(batch) >= args.batch_size:
                flush()

            processed_count += 1

            if processed_count % args.batch_save == 0 or processed_count == len(items):
                flush()
                elapsed = time.time() - start_time
                avg_speed = elapsed / max(1, processed_count - (len(items) - len(pending_items)))
                print(
                    f"Progress: [{processed_count}/{len(items)}] | "
                    f"Avg speed: {avg_speed:.2f}s/query | "
                    f"Elapsed: {elapsed / 60:.1f}m"
                )
                atomic_write_json(predictions_path, predictions)
                atomic_write_json(
                    checkpoint_path,
                    {"metadata": metadata, "predictions": predictions},
                )
                if processed_count % (args.batch_save * 2) == 0:
                    torch.cuda.empty_cache()
        flush()

    print(f"\nInference finished for {len(predictions)} queries. Saved to {predictions_path}")

    # Metrics only exist when the dataset carries ground-truth bboxes (e.g. val).
    has_ground_truth = bool(items) and "bbox" in items[0]
    metrics = evaluate_predictions(items, predictions)

    if metrics is not None:
        print("\n" + "="*50)
        print("=== EVALUATION REPORT (GROUND TRUTH DETECTED) ===")
        print(f"Total Evaluated Samples : {metrics['total']}")
        print(f"Accuracy @ IoU >= 0.5   : {metrics['acc_at_0_5'] * 100:.2f}% "
              f"({metrics['hits']}/{metrics['total']})")
        print(f"Mean IoU (mIoU)         : {metrics['mean_iou']:.4f}")
        print(f"Failed Bbox Predictions : {metrics['failures']}")
        print("="*50 + "\n")

    # Persist Modal-compatible artifacts: metadata.json + checkpoint.json + summary.json
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(
        checkpoint_path,
        {"metadata": metadata, "predictions": predictions},
    )
    atomic_write_json(
        run_dir / "summary.json",
        {
            "metadata": metadata,
            "metrics": metrics,
            "total_predictions": len(predictions),
            "valid_predictions": sum(
                validate_bbox(value) is not None for value in predictions.values()
            ),
            "submission_ready": False,
        },
    )
    print(f"Metadata saved to: {metadata_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Summary saved to: {run_dir / 'summary.json'}")

    # If this is a full test run without ground truth, package submission.zip
    official_template_path = paths.submission_template
    if (
        not has_ground_truth
        and args.limit == 0
        and len(predictions) == len(items)
        and official_template_path.is_file()
    ):
        print("Packaging competition submission ZIP...")
        build_submission(
            test_json_path=official_template_path,
            predictions_path=predictions_path,
            output_dir=run_dir,
        )


if __name__ == "__main__":
    main()
