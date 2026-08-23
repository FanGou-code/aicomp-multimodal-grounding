"""Standalone single-GPU offline inference and evaluation entry.

Model-agnostic: ``--model`` selects a grounding adapter (qwen3vl /
internvl35 / groundingdino / mock). Handles both:
1. Competition Submission Generation (when running on unannotated test queries)
2. Validation Accuracy & IoU Score Evaluation (when running on annotated val/train queries)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import multiprocessing
from pathlib import Path
import sys
import time

# This entrypoint lives in offline/; make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.inference_core import evaluate_predictions, load_inference_items
from aicomp_grounding.inference_state import (
    assign_pending_shards,
    build_shard_metadata,
    build_run_metadata,
    fingerprint_inputs,
    fingerprint_lora,
    validate_checkpoint_payload,
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
        "--num-shards",
        type=int,
        default=1,
        help="Number of local processes to split inference across.",
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
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for CPU/GPU overlap. 0 = sequential (legacy).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root; relative input and output paths resolve from here.",
    )
    parser.add_argument("--shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-root", type=Path, default=None, help=argparse.SUPPRESS)
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


def _run_inference_loop(
    items: list[dict],
    adapter,
    *,
    adapter_dir: Path | None,
    model_path: str | None,
    data_dir: Path,
    batch_size: int,
    batch_save: int,
    existing_predictions: dict[str, list[float] | None] | None = None,
    device: str = "cuda",
    checkpoint_path: Path | None = None,
    checkpoint_metadata: dict | None = None,
) -> dict[str, list[float] | None]:
    """Run the existing sequential batch inference loop and return predictions."""
    predictions: dict[str, list[float] | None] = dict(existing_predictions or {})
    pending_items = [item for item in items if item["key"] not in predictions]
    if pending_items:
        pending_items.sort(key=lambda x: (x.get("visible", x["key"]), x["key"]))
        print(f"Loading model '{adapter.model_name}' (revision {adapter.model_revision})...")
        adapter.load(device=device, lora_path=adapter_dir, model_path=model_path)

        scene_cache: OrderedDict = OrderedDict()
        batch: list[ModelInput] = []

        def flush() -> None:
            if not batch:
                return
            results = adapter.predict(batch)
            for sample, result in zip(batch, results):
                predictions[sample.key] = result.bbox
            batch.clear()

        start_time = time.time()
        processed_count = len(predictions)
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
                images = load_scene_images(item, data_dir)
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
            if len(batch) >= batch_size:
                flush()

            processed_count += 1
            if processed_count % batch_save == 0 or processed_count == len(items):
                flush()
                elapsed = time.time() - start_time
                avg_speed = elapsed / max(1, processed_count - (len(items) - len(pending_items)))
                print(
                    f"Progress: [{processed_count}/{len(items)}] | "
                    f"Avg speed: {avg_speed:.2f}s/query | "
                    f"Elapsed: {elapsed / 60:.1f}m"
                )
                if checkpoint_path is not None and checkpoint_metadata is not None:
                    atomic_write_json(
                        checkpoint_path,
                        {"metadata": checkpoint_metadata, "predictions": predictions},
                    )
        flush()
    return predictions


def _run_dataloader_inference_loop(
    items: list[dict],
    adapter,
    *,
    adapter_dir: Path | None,
    model_path: str | None,
    data_dir: Path,
    batch_size: int,
    batch_save: int,
    num_workers: int,
    existing_predictions: dict[str, list[float] | None] | None = None,
    device: str = "cuda",
    checkpoint_path: Path | None = None,
    checkpoint_metadata: dict | None = None,
) -> dict[str, list[float] | None]:
    """DataLoader-based inference loop for CPU/GPU overlap."""
    from torch.utils.data import DataLoader, Dataset

    predictions: dict[str, list[float] | None] = dict(existing_predictions or {})
    pending_items = [item for item in items if item["key"] not in predictions]
    if not pending_items:
        return predictions

    pending_items.sort(key=lambda x: (x.get("visible", x["key"]), x["key"]))
    print(f"Loading model '{adapter.model_name}' (revision {adapter.model_revision})...")
    adapter.load(device=device, lora_path=adapter_dir, model_path=model_path)
    use_prepared = getattr(adapter, "supports_prepared_inputs", False)

    class _Dataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            item = self.items[idx]
            visible_img, infrared_img, depth_img = load_scene_images(item, data_dir)
            return {
                "key": item["key"],
                "visible": visible_img,
                "infrared": infrared_img,
                "depth": depth_img,
                "query": item["query"],
            }

    def _collate(batch):
        samples = [
            ModelInput(
                visible=item["visible"],
                infrared=item["infrared"],
                depth=item["depth"],
                query=item["query"],
                key=item["key"],
            )
            for item in batch
        ]
        if use_prepared:
            # collate_fn runs inside the DataLoader worker processes, so this
            # moves CPU-side prompt building and image processing off the
            # main loop, overlapping them with GPU generation.
            return [s.key for s in samples], adapter.prepare_inputs(samples)
        return [s.key for s in samples], samples

    dataset = _Dataset(pending_items)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate,
    )

    start_time = time.time()
    processed_count = len(predictions)
    since_last_save = 0

    for keys, payload in loader:
        results = (
            adapter.predict_from_inputs(payload)
            if use_prepared
            else adapter.predict(payload)
        )
        for key, result in zip(keys, results):
            predictions[key] = result.bbox
        processed_count += len(keys)
        since_last_save += len(keys)

        if since_last_save >= batch_save or processed_count == len(items):
            elapsed = time.time() - start_time
            speed = processed_count / max(1e-5, elapsed)
            print(
                f"Progress: [{processed_count}/{len(items)}] | "
                f"Speed: {speed:.2f} samples/s | "
                f"Elapsed: {elapsed / 60:.1f}m"
            )
            if checkpoint_path is not None and checkpoint_metadata is not None:
                atomic_write_json(
                    checkpoint_path,
                    {"metadata": checkpoint_metadata, "predictions": predictions},
                )
            since_last_save = 0

    return predictions


def _run_shard_worker(
    payload: tuple[dict, list[dict], int, Path | None, dict],
):
    """Spawnable worker used by local multiprocess inference."""
    args_dict, items, shard_id, checkpoint_dir, shard_metadata = payload
    args = argparse.Namespace(**args_dict)
    adapter_kwargs = {}
    if args.model in ("qwen3vl", "qwen3vl32"):
        adapter_kwargs["max_pixels"] = args.max_pixels
    adapter = get_adapter(args.model, **adapter_kwargs)
    adapter_dir = Path(args.lora_path).resolve() if args.lora_path else None
    if adapter_dir is not None and not adapter_dir.exists():
        adapter_dir = None
    if adapter_dir is not None and not adapter.supports_lora:
        adapter_dir = None
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        num_gpus = 0
    device = f"cuda:{shard_id % num_gpus}" if num_gpus > 1 else "cuda"
    checkpoint_path = (
        Path(checkpoint_dir) / f"shard_{shard_id}.checkpoint.json"
        if checkpoint_dir is not None
        else None
    )
    if getattr(args, "num_workers", 0) > 0:
        predictions = _run_dataloader_inference_loop(
            items,
            adapter,
            adapter_dir=adapter_dir,
            model_path=args.model_path,
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            batch_save=args.batch_save,
            num_workers=args.num_workers,
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=shard_metadata,
        )
    else:
        predictions = _run_inference_loop(
            items,
            adapter,
            adapter_dir=adapter_dir,
            model_path=args.model_path,
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            batch_save=args.batch_save,
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=shard_metadata,
        )
    return shard_id, predictions


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
    if args.model in ("qwen3vl", "qwen3vl32"):
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
        num_shards=args.num_shards,
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
    if args.resume:
        if predictions_path.is_file():
            print(f"Resuming from existing predictions file: {predictions_path}")
            predictions = load_json(predictions_path)
        elif checkpoint_path.is_file():
            print(f"Resuming from existing checkpoint file: {checkpoint_path}")
            ckpt_data = load_json(checkpoint_path)
            predictions = ckpt_data.get("predictions", {}) if isinstance(ckpt_data, dict) else {}
    print(f"Total queries in dataset: {len(items)} | Already finished: {len(predictions)} | Pending: {len(items) - len(predictions)}")
    if args.num_shards > 1:
        shard_checkpoint_dir = run_dir / "shard_checkpoints"
        shard_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shards = assign_pending_shards(
            items,
            num_shards=args.num_shards,
            existing_predictions=predictions,
        )
        worker_payloads = [
            (
                dict(vars(args)),
                shard,
                shard_id,
                shard_checkpoint_dir,
                build_shard_metadata(
                    metadata,
                    shard_id,
                    [item["key"] for item in shard],
                ),
            )
            for shard_id, shard in enumerate(shards)
        ]
        context = multiprocessing.get_context("spawn")
        with context.Pool(len(shards)) as pool:
            shard_results = pool.map(_run_shard_worker, worker_payloads)
        for _, shard_predictions in shard_results:
            # Recover shard_id and metadata from the result is not necessary;
            # validation is done below against the same shard assignments.
            for key, value in shard_predictions.items():
                if key in predictions:
                    raise ValueError(f"Duplicate prediction key across shards: {key!r}")
                predictions[key] = value
        for shard_id, shard in enumerate(shards):
            shard_metadata = build_shard_metadata(
                metadata,
                shard_id,
                [item["key"] for item in shard],
            )
            shard_predictions = {
                key: predictions[key]
                for key in [item["key"] for item in shard]
            }
            validate_checkpoint_payload(
                {
                    "metadata": shard_metadata,
                    "predictions": shard_predictions,
                },
                shard_metadata,
                [item["key"] for item in shard],
                require_complete=True,
                label=f"local shard {shard_id}",
            )
    else:
        if args.num_workers > 0:
            predictions = _run_dataloader_inference_loop(
                items,
                adapter,
                adapter_dir=adapter_dir,
                model_path=args.model_path,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                batch_save=args.batch_save,
                num_workers=args.num_workers,
                existing_predictions=predictions,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata=metadata,
            )
        else:
            predictions = _run_inference_loop(
                items,
                adapter,
                adapter_dir=adapter_dir,
                model_path=args.model_path,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                batch_save=args.batch_save,
                existing_predictions=predictions,
            )
    atomic_write_json(predictions_path, predictions)
    atomic_write_json(
        checkpoint_path,
        {"metadata": metadata, "predictions": predictions},
    )

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
            allow_fallback=True,
        )


if __name__ == "__main__":
    main()
