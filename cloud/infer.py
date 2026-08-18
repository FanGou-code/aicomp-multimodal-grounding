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

from aicomp_grounding.config import (
    DATA_ROOT,
    MAX_PIXELS,
    MIN_PIXELS,
    MODAL_GPU_PACKAGES,
    MODEL_NAME,
    MODEL_REVISION,
)
from aicomp_grounding.inference_core import (
    evaluate_predictions,
    load_inference_items,
    merge_shard_results,
)

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
) -> dict[str, Any]:
    """Run Batch-4 high-throughput visual grounding inference on a shard of queries."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from peft import PeftModel
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from aicomp_grounding.bbox import compute_iou, parse_bbox_from_text
    from aicomp_grounding.prompts import build_grounding_messages

    dataset_volume.reload()
    data_root = Path(DATA_ROOT)

    total_samples = len(items)
    print(f"Loading Base Model: {MODEL_NAME} on H100 for {total_samples} queries...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    processor.tokenizer.padding_side = "left"

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")

    if adapter_path and os.path.exists(adapter_path):
        print(f"Attaching LoRA Adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path).to("cuda")
    else:
        print("Running in Base Model mode (no LoRA adapter attached)...")
        model = base_model
    model.eval()

    class GroundingDataset(Dataset):
        def __init__(self, sample_items, root_dir):
            self.items = sample_items
            self.root_dir = root_dir

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            key = item.get("key", str(idx))
            gt_bbox = item.get("bbox", None)

            vis_path = self.root_dir / item["visible"]
            ir_path = self.root_dir / item["infrared"]
            dp_path = self.root_dir / item["depth"]

            with Image.open(vis_path) as im:
                vis_img = im.convert("RGB")
            with Image.open(ir_path) as im:
                ir_img = im.convert("RGB")
            with Image.open(dp_path) as im:
                dp_img = im.convert("RGB")

            messages = build_grounding_messages(vis_img, ir_img, dp_img, item["query"])
            return {
                "key": key,
                "messages": messages,
                "gt_bbox": gt_bbox,
            }

    dataset = GroundingDataset(items, data_root)

    def collate_batch(batch_items):
        keys = [item["key"] for item in batch_items]
        messages_list = [item["messages"] for item in batch_items]
        gt_bboxes = [item["gt_bbox"] for item in batch_items]

        texts = [
            processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return keys, inputs, gt_bboxes

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_batch,
        pin_memory=True,
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
        for keys, batch_inputs, gt_bboxes in loader:
            model_inputs = {
                k: v.to("cuda", non_blocking=True)
                for k, v in batch_inputs.items()
                if torch.is_tensor(v)
            }

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=32,
                    do_sample=False,
                )

            prompt_len = model_inputs["input_ids"].shape[1]
            generated_tokens = generated_ids[:, prompt_len:]
            text_outputs = processor.batch_decode(
                generated_tokens, skip_special_tokens=False
            )

            for key, text_out, gt_bbox in zip(keys, text_outputs, gt_bboxes):
                pred_bbox = parse_bbox_from_text(text_out)
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
):
    """Local entrypoint for running validation or test inference on Modal."""
    from aicomp_grounding.submission import build_submission

    print(f"=================================================================")
    print(f"🚀 Modal Visual Grounding Inference Engine")
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
            )
        )
    else:
        results = [
            run_shard_inference.remote(
                items=items,
                adapter_path=adapter_path,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        ]

    # Merge results from all shards (wall clock = slowest shard)
    merged = merge_shard_results(results)
    all_predictions = merged["predictions"]
    total_time = merged["elapsed_seconds"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    predictions_file = out_path / f"predictions_{split}.json"
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
