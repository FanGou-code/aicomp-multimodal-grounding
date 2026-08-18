"""Official High-Speed Modal Multi-Worker Batch-Inference Engine for Qwen3-VL.

Supports:
1. Validation Set Evaluation (--split val): Computes ACC@0.5 and mIoU.
2. Official Test Set Generation (--split test): Runs high-speed Batch-4 inference on 9,555 test queries
   with automatic packaging into a verified official submission.zip.
3. Multi-Card Elastic Sharding (--num-shards 8): Automatically distributes work across parallel H100s.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import time
from typing import Any

import modal

# This entrypoint lives in cloud/; make the repository root importable so the
# shared library and the scripts package resolve under `modal run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.config import DATA_ROOT, MODAL_GPU_PACKAGES
from aicomp_grounding.inference_core import (
    evaluate_predictions,
    load_inference_items,
    merge_shard_results,
)
from aicomp_grounding.models import available_models

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(*MODAL_GPU_PACKAGES)
    .add_local_python_source("aicomp_grounding")
)

app = modal.App("rgbdt-modal-inference", image=image)

dataset_volume = modal.Volume.from_name("rgbdt-dataset")
model_volume = modal.Volume.from_name("hf-model-cache")


@app.function(
    gpu="H100",
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={
        "/data": dataset_volume,
        "/root/.cache/huggingface": model_volume,
    },
)
def run_shard_inference(
    items: list[dict],
    adapter_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    model: str = "qwen3vl",
) -> dict[str, Any]:
    """Run high-throughput grounding inference on a shard of queries.

    Model-agnostic: ``model`` selects the grounding adapter; the DataLoader
    pipeline only loads tri-modal images, all model-specific preprocessing
    and generation live inside the adapter.
    """
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from aicomp_grounding.bbox import compute_iou
    from aicomp_grounding.models import get_adapter
    from aicomp_grounding.models.base import ModelInput

    dataset_volume.reload()
    data_root = Path(DATA_ROOT)

    total_samples = len(items)
    adapter = get_adapter(model)
    print(
        f"Loading model '{adapter.model_name}' (rev {adapter.model_revision}) "
        f"on H100 for {total_samples} queries..."
    )

    lora_path = None
    if adapter_path and os.path.exists(adapter_path):
        if adapter.supports_lora:
            print(f"Attaching LoRA Adapter: {adapter_path}...")
            lora_path = Path(adapter_path)
        else:
            print(f"Warning: --model {adapter.name} does not use LoRA; ignoring adapter path.")
    else:
        print("Running in Base Model mode (no LoRA adapter attached)...")

    adapter.load(device="cuda", lora_path=lora_path)

    class GroundingDataset(Dataset):
        """Generic: worker-side work is tri-modal image loading only."""

        def __init__(self, sample_items, root_dir):
            self.items = sample_items
            self.root_dir = root_dir

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx) -> dict:
            item = self.items[idx]
            with Image.open(self.root_dir / item["visible"]) as im:
                vis_img = im.convert("RGB")
            with Image.open(self.root_dir / item["infrared"]) as im:
                ir_img = im.convert("RGB")
            with Image.open(self.root_dir / item["depth"]) as im:
                dp_img = im.convert("RGB")
            return {
                "key": item.get("key", str(idx)),
                "visible": vis_img,
                "infrared": ir_img,
                "depth": dp_img,
                "query": item["query"],
                "gt_bbox": item.get("bbox", None),
            }

    dataset = GroundingDataset(items, data_root)

    def collate_batch(batch_items):
        samples = [
            ModelInput(
                visible=item["visible"],
                infrared=item["infrared"],
                depth=item["depth"],
                query=item["query"],
                key=item["key"],
            )
            for item in batch_items
        ]
        keys = [item["key"] for item in batch_items]
        gt_bboxes = [item["gt_bbox"] for item in batch_items]
        return samples, keys, gt_bboxes

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_batch,
    )

    print(
        f"\n🚀 Running Pipeline (Batch={batch_size}, {num_workers} Workers) on {total_samples} samples..."
    )
    start_time = time.time()

    predictions: dict[str, list[float] | None] = {}
    correct_05 = 0
    total_iou = 0.0
    has_ground_truth = any(item.get("bbox") is not None for item in items)
    processed = 0

    with torch.no_grad():
        for samples, keys, gt_bboxes in loader:
            results = adapter.predict(samples)

            for key, result, gt_bbox in zip(keys, results, gt_bboxes):
                pred_bbox = result.bbox
                if pred_bbox is not None and gt_bbox is not None:
                    iou = compute_iou(pred_bbox, gt_bbox)
                    total_iou += iou
                    if iou >= 0.5:
                        correct_05 += 1
                predictions[key] = pred_bbox

            processed += len(keys)
            if processed % 40 == 0 or processed == total_samples:
                elapsed = time.time() - start_time
                speed = processed / max(1e-5, elapsed)
                if has_ground_truth:
                    acc = (correct_05 / processed) * 100
                    miou = (total_iou / processed) * 100
                    print(
                        f"[{processed:4d}/{total_samples}] ACC@0.5: {acc:5.2f}% | mIoU: {miou:5.2f}% | "
                        f"Speed: {speed:4.2f} s/s | Elapsed: {elapsed:5.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[{processed:4d}/{total_samples}] Speed: {speed:4.2f} samples/s | Elapsed: {elapsed:5.1f}s",
                        flush=True,
                    )

    elapsed_total = time.time() - start_time
    return {
        "predictions": predictions,
        "elapsed_seconds": elapsed_total,
        "correct_05": correct_05 if has_ground_truth else None,
        "total_iou": total_iou if has_ground_truth else None,
        "total_samples": total_samples,
    }


@app.local_entrypoint()
def main(
    split: str = "val",
    adapter_path: str = f"{DATA_ROOT}/output_lora/train_9e468a454061153b/best/epoch_02",
    annotation_run_id: str = "annot_ac72f1d926bb2d23",
    batch_size: int = 4,
    num_workers: int = 4,
    num_shards: int = 1,
    output_dir: str = "outputs/modal_inference",
    model: str = "qwen3vl",
):
    """Local entrypoint for running validation or test inference on Modal."""
    from aicomp_grounding.submission import build_submission

    if model not in available_models():
        raise ValueError(f"Unknown model {model!r}; available: {available_models()}")

    print(f"=================================================================")
    print(f"🚀 Modal Visual Grounding Inference Engine")
    print(f"  Model:              {model}")
    print(f"  Split:              {split}")
    print(f"  Adapter:            {adapter_path}")
    print(f"  Batch Size:         {batch_size}")
    print(f"  Parallel Shards:    {num_shards}")
    print(f"=================================================================")

    # Load dataset index from local data dir; path policy stays platform
    # specific, item normalization is shared via the inference core.
    if split == "val":
        data_root = Path("data")
        val_approved_path = (
            data_root
            / "outputs"
            / "annotations"
            / annotation_run_id
            / "val"
            / "approved.json"
        )
        if not val_approved_path.exists():
            # Fallback to standard val.json
            val_approved_path = data_root / "val.json"
        if not val_approved_path.exists():
            raise FileNotFoundError(f"Val dataset not found at {val_approved_path}")
        index_path = val_approved_path
    else:  # test split
        test_template_path = Path("data/Test/queries/queries.json")
        if not test_template_path.exists():
            test_template_path = Path("data/test.json")
        if not test_template_path.exists():
            raise FileNotFoundError(f"Test queries not found at {test_template_path}")
        index_path = test_template_path

    items, _ = load_inference_items(index_path)

    total_queries = len(items)
    print(f"Loaded {total_queries} queries for split '{split}'.")

    # Shard items across parallel workers
    if num_shards > 1:
        shards = [items[i::num_shards] for i in range(num_shards)]
        print(f"Dispatching {total_queries} queries across {num_shards} parallel H100 containers...")
        results = list(
            run_shard_inference.map(
                shards,
                [adapter_path] * num_shards,
                [batch_size] * num_shards,
                [num_workers] * num_shards,
                [model] * num_shards,
            )
        )
    else:
        results = [
            run_shard_inference.remote(
                items=items,
                adapter_path=adapter_path,
                batch_size=batch_size,
                num_workers=num_workers,
                model=model,
            )
        ]

    # Merge results from all shards (wall clock = slowest shard)
    merged = merge_shard_results(results)
    all_predictions = merged["predictions"]
    total_time = merged["elapsed_seconds"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    predictions_file = out_path / f"predictions_{model}_{split}.json"
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"🏆 INFERENCE COMPLETED SUCCESSFULLY")
    print(f"  Total Queries:      {total_queries}")
    print(f"  Predictions Saved:  {predictions_file}")
    print(f"  Effective Runtime:  {total_time:.1f}s ({total_time/60:.2f} mins)")
    print(f"  Global Throughput:  {total_queries/max(1e-5, total_time):.2f} samples/s")

    if split == "val" and total_queries > 0:
        metrics = evaluate_predictions(items, all_predictions)
        if metrics is not None:
            print(
                f"  ACC@0.5 Accuracy:   {metrics['acc_at_0_5'] * 100:.4f}% "
                f"({metrics['hits']}/{metrics['total']})"
            )
            print(f"  Mean IoU:           {metrics['mean_iou'] * 100:.4f}%")

    if split == "test":
        print("\n📦 Generating verified official submission ZIP...")
        zip_path = build_submission(
            test_json_path=test_template_path,
            predictions_path=predictions_file,
            output_dir=out_path,
            allow_fallback=False,
        )
        print(f"✅ Submission Ready: {zip_path}")
    print("=" * 65 + "\n")
