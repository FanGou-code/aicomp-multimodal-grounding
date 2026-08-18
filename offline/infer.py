"""Standalone single-GPU offline inference and evaluation script for Qwen3-VL.

Handles both:
1. Competition Submission Generation (when running on unannotated test queries)
2. Validation Accuracy & IoU Score Evaluation (when running on annotated val/train queries)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys
import time

import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# This entrypoint lives in offline/; make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.bbox import parse_bbox_from_text, validate_bbox
from aicomp_grounding.config import (
    INFERENCE_COMPUTE_DTYPE,
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_NAME,
    MODEL_REVISION,
)
from aicomp_grounding.inference_core import evaluate_predictions, load_inference_items
from aicomp_grounding.inference_state import (
    build_run_metadata,
    fingerprint_inputs,
    fingerprint_lora,
)
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.prompts import (
    build_grounding_messages,
    grounding_prompt_hash,
    GROUNDING_SYSTEM_PROMPT,
)
from aicomp_grounding.submission import build_submission

CACHE_MAX_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline Qwen3-VL inference or evaluation on a single GPU."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=Path("model/Qwen3-VL-8B-Instruct"),
        help="Path to local Qwen3-VL-8B-Instruct base model (downloaded offline).",
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
        help="Processor max_pixels override. Training uses 3072 patches "
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
        default=Path("best/epoch_02"),
        help="Path to downloaded LoRA adapter directory (e.g. best/epoch_02).",
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
    return parser.parse_args()


def resolve_image_path(raw_path: str, data_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (data_dir / p).resolve()


def load_scene_images(item: dict, data_dir: Path) -> tuple[Image.Image, Image.Image, Image.Image]:
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

    if not args.test_json.is_file():
        raise FileNotFoundError(f"Dataset JSON index not found: {args.test_json}")

    # Accepts an approved annotation artifact or a flat {query_id: item}
    # index; see the shared inference core for the exact contract.
    items, approved_metadata = load_inference_items(args.test_json, limit=args.limit)

    # ---- Build a Modal-compatible run identity ----------------------------
    selected_keys = [item["key"] for item in items]
    adapter_dir = Path(args.lora_path).resolve() if args.lora_path.exists() else None
    adapter_fingerprint = fingerprint_lora(adapter_dir)
    dataset_for_fp = {item["key"]: item for item in items}
    input_fingerprint = fingerprint_inputs(dataset_for_fp, selected_keys)
    prompt_hash = grounding_prompt_hash(GROUNDING_SYSTEM_PROMPT)
    generation_config = {
        "max_new_tokens": 32,
        "do_sample": False,
        "max_pixels": args.max_pixels,
    }

    metadata = build_run_metadata(
        mode="base",
        split="test" if approved_metadata is None else approved_metadata["split"],
        annotation_run_id=args.annotation_run_id,
        model=MODEL_NAME,
        model_revision=MODEL_REVISION,
        lora_path=adapter_dir,
        adapter_fingerprint=adapter_fingerprint,
        prompt_hash=prompt_hash,
        generation_config=generation_config,
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

        model_source = args.model_path
        print(f"Loading base model from '{model_source}' ({INFERENCE_COMPUTE_DTYPE})...")
        compute_dtype = getattr(torch, INFERENCE_COMPUTE_DTYPE)

        processor_kwargs = {
            "min_pixels": MIN_PIXELS,
            "max_pixels": args.max_pixels,
        }
        if not os.path.exists(model_source):
            processor_kwargs["revision"] = MODEL_REVISION

        processor = AutoProcessor.from_pretrained(
            model_source,
            **processor_kwargs,
        )

        model_kwargs = {
            "dtype": compute_dtype,
            "attn_implementation": "sdpa",
            "device_map": "auto",
        }

        if not os.path.exists(model_source):
            model_kwargs["revision"] = MODEL_REVISION

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source,
            **model_kwargs,
        )

        print(f"Attaching LoRA adapter from '{args.lora_path}'...")
        model = PeftModel.from_pretrained(base_model, str(args.lora_path))
        model.eval()

        device = next(model.parameters()).device

        scene_cache: OrderedDict = OrderedDict()

        start_time = time.time()
        processed_count = len(predictions)

        print("Starting single-GPU inference...")

        with torch.no_grad():
            for item in pending_items:
                key = item["key"]
                query = item["query"]

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

                messages = build_grounding_messages(images[0], images[1], images[2], query)
                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[prompt_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False,
                    )

                prompt_len = inputs["input_ids"].shape[1]
                generated_tokens = generated_ids[0][prompt_len:]
                text_output = processor.decode(
                    generated_tokens, skip_special_tokens=False
                )

                bbox = parse_bbox_from_text(text_output)

                predictions[key] = bbox
                processed_count += 1

                if processed_count % args.batch_save == 0 or processed_count == len(items):
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
    official_template_path = args.data_dir / "Test/queries/queries.json"
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
