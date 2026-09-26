"""Standalone single-GPU offline inference and evaluation entry.

Model-agnostic: ``--model`` selects a grounding adapter (qwen3vl / qwen3_5 /
glm46v / mock).

This entrypoint produces predictions only. Building ``submission.zip`` is a
separate, explicit step (``python tools/submission.py``); inference
never packages one.
"""

from __future__ import annotations

import argparse
import functools
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable
from datetime import datetime
import multiprocessing
from pathlib import Path
import sys
import time

# Make the repository root importable when executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.images import trusted_dataset_image_fingerprint
from aicomp_grounding.grounding.engine.inference_core import evaluate_predictions, load_inference_items
from aicomp_grounding.grounding.engine.inference_state import (
    assign_pending_shards,
    build_shard_metadata,
    build_run_metadata,
    fingerprint_inputs,
    fingerprint_lora,
    load_resume_predictions,
    validate_checkpoint_payload,
)
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.grounding.models import available_models, get_adapter
from aicomp_grounding.grounding.models.base import ModelInput, require_local_model_path
from aicomp_grounding.config import (
    INFERENCE_DEFAULT_MAX_PIXELS as MAX_PIXELS,
    load_run_config,
    merge_run_config,
)
from aicomp_grounding.paths import ProjectPaths, output_dir, resolve_from_root

#: Adapters whose constructor takes the per-frame pixel budget (``max_pixels``).
#: ``mock`` is not in this set: it has no such parameter, and passing it would
#: raise TypeError.  This is a capability set, not the training roster — a future
#: inference-only VLM belongs here without being trainable.
PIXEL_BUDGET_MODELS = ("qwen3vl", "qwen3_5", "glm46v")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_eta(seconds: int) -> str:
    if seconds < 0:
        return "-"
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


#: Keys accepted in the ``run:`` section of a --config YAML.
CONFIG_RUN_KEYS = (
    "model",
    "model_path",
    "test_json",
    "max_pixels",
    "data_dir",
    "lora_path",
    "batch_size",
    "num_shards",
    "output_dir",
    "annotation_run_id",
    "run_tag",
    "limit",
    "resume",
    "batch_save",
    "num_workers",
    "project_root",
)

#: Inference has no training hyperparameters; a ``hyperparameters`` section is
#: rejected so a training config is never silently accepted here.
CONFIG_HYPERPARAMETER_KEYS: tuple[str, ...] = ()

#: Applied after CLI and YAML; ``None`` means "not specified yet".
_RUN_DEFAULTS = {
    "model": "qwen3vl",
    "test_json": Path("data/Test/queries/queries.json"),
    "max_pixels": MAX_PIXELS,
    "data_dir": Path("data"),
    "batch_size": 1,
    "num_shards": 1,
    "output_dir": output_dir("inference"),
    "annotation_run_id": "",
    "run_tag": "",
    "limit": 0,
    "resume": True,
    "batch_save": 50,
    "num_workers": 4,
    "project_root": Path("."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline grounding inference or evaluation on a single GPU."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML run configuration; CLI flags override it.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=available_models(),
        help="Grounding model adapter to run (default: qwen3vl).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Pre-downloaded model directory (required except for mock). No automatic download.",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=None,
        help=(
            "Path to input dataset JSON index; accepts approved artifacts "
            "or the official Test template directly "
            "(default: data/Test/queries/queries.json)."
        ),
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Single-frame processor max_pixels override (default: 3072*28*28); "
        "this is also the value recorded in the inference run identity. "
        "Lower it (e.g. 1920*28*28=1505280) if the inference GPU is short on "
        "VRAM. Lowering hurts grounding accuracy.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to root data directory containing Test/ (default: data).",
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
        default=None,
        help="Number of queries predicted per adapter call (default: 1).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of local processes to split inference across (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for inference outputs; results go to <output-dir>/<run_id>/ "
        "(default: outputs/inference).",
    )
    parser.add_argument(
        "--annotation-run-id",
        type=str,
        default=None,
        help="Annotation run id when --test-json is an approved artifact (train/val).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Arbitrary experiment tag folded into the run id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If > 0, limit execution to the first N queries (default: 0).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from existing predictions.json or checkpoint.json in output-dir if present (default: on).",
    )
    parser.add_argument(
        "--batch-save",
        type=int,
        default=None,
        help="Save predictions.json checkpoint every N queries (default: 50).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers for CPU/GPU overlap (default: 4). 0 = sequential.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root; relative input and output paths resolve from here.",
    )
    parser.add_argument("--shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config is not None:
        merge_run_config(
            args,
            load_run_config(args.config),
            parser=parser,
            run_keys=CONFIG_RUN_KEYS,
            hyperparameter_keys=CONFIG_HYPERPARAMETER_KEYS,
        )
    for key, value in _RUN_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def resolve_image_path(raw_path: str, data_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (data_dir / p).resolve()


@functools.lru_cache(maxsize=32)
def _cached_read_image(v_path: str):
    from PIL import Image

    return Image.open(v_path).convert("RGB")


def load_query_image(item: dict, data_dir: Path):
    """Decode the visible RGB image referenced by an inference item."""
    images_dict = item.get("images") if isinstance(item.get("images"), dict) else item
    v_raw = images_dict.get("visible")

    if not v_raw:
        raise ValueError(f"Sample item missing visible image path: {item}")

    v_path = resolve_image_path(v_raw, data_dir)

    if not v_path.is_file():
        raise FileNotFoundError(f"Visible image not found: {v_path}")

    return _cached_read_image(str(v_path)).copy()


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
    commit_hook: Callable[[], None] | None = None,
) -> dict[str, list[float] | None]:
    """Run the existing sequential batch inference loop and return predictions."""
    predictions: dict[str, list[float] | None] = dict(existing_predictions or {})
    pending_items = [item for item in items if item["key"] not in predictions]
    if pending_items:
        pending_items.sort(key=lambda x: (x.get("visible", x["key"]), x["key"]))
        print(f"Loading model '{adapter.model_name}' (revision {adapter.model_revision})...")
        adapter.load(device=device, lora_path=adapter_dir, model_path=model_path)

        batch: list[ModelInput] = []

        def flush() -> None:
            if not batch:
                return
            results = adapter.predict(batch)
            for sample, result in zip(batch, results, strict=True):
                predictions[sample.key] = result.bbox
            batch.clear()

        start_time = time.time()
        initial_count = len(predictions)
        processed_count = initial_count
        for item in pending_items:
            batch.append(
                ModelInput(
                    visible=load_query_image(item, data_dir),
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
                session_processed = processed_count - initial_count
                speed = session_processed / max(1e-5, elapsed)
                remaining = max(len(items) - processed_count, 0)
                eta = _format_eta(int(remaining / max(speed, 1e-6)))
                percent = 100.0 * processed_count / len(items)
                print(
                    f"[{_now_str()}] Progress {processed_count}/{len(items)} "
                    f"({percent:.1f}%) | "
                    f"Speed: {speed:.2f} samples/s | "
                    f"ETA: {eta}"
                )
                if checkpoint_path is not None and checkpoint_metadata is not None:
                    atomic_write_json(
                        checkpoint_path,
                        {
                            "metadata": checkpoint_metadata,
                            "predictions": predictions,
                            "assigned_keys": [item["key"] for item in items],
                        },
                    )
                    if commit_hook is not None:
                        commit_hook()
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
    commit_hook: Callable[[], None] | None = None,
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
    # The model is loaded BEFORE the DataLoader spawns workers, which relies on
    # Linux fork semantics (workers never touch CUDA). Spawning DataLoader
    # workers would try to pickle the adapter incl. the CUDA model and fail.
    use_prepared = getattr(adapter, "supports_prepared_inputs", False)

    class _Dataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            item = self.items[idx]
            return {
                "key": item["key"],
                "visible": load_query_image(item, data_dir),
                "query": item["query"],
            }

    def _collate(batch):
        samples = [
            ModelInput(
                visible=item["visible"],
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
        multiprocessing_context=multiprocessing.get_context("fork") if num_workers > 0 else None,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate,
    )

    start_time = time.time()
    initial_count = len(predictions)
    processed_count = initial_count
    since_last_save = 0

    for keys, payload in loader:
        results = (
            adapter.predict_from_inputs(payload)
            if use_prepared
            else adapter.predict(payload)
        )
        for key, result in zip(keys, results, strict=True):
            predictions[key] = result.bbox
        processed_count += len(keys)
        since_last_save += len(keys)

        if since_last_save >= batch_save or processed_count == len(items):
            elapsed = time.time() - start_time
            session_processed = processed_count - initial_count
            speed = session_processed / max(1e-5, elapsed)
            remaining = max(len(items) - processed_count, 0)
            eta = _format_eta(int(remaining / max(speed, 1e-6)))
            percent = 100.0 * processed_count / len(items)
            print(
                f"[{_now_str()}] Progress {processed_count}/{len(items)} "
                f"({percent:.1f}%) | "
                f"Speed: {speed:.2f} samples/s | "
                f"ETA: {eta}"
            )
            if checkpoint_path is not None and checkpoint_metadata is not None:
                atomic_write_json(
                    checkpoint_path,
                    {
                        "metadata": checkpoint_metadata,
                        "predictions": predictions,
                        "assigned_keys": [item["key"] for item in items],
                    },
                )
                if commit_hook is not None:
                    commit_hook()
            since_last_save = 0

    return predictions


def _run_shard_worker(
    payload: tuple[dict, list[dict], int, Path | None, dict],
):
    """Spawnable worker used by local multiprocess inference."""
    args_dict, items, shard_id, checkpoint_dir, shard_metadata = payload
    args = argparse.Namespace(**args_dict)
    adapter_kwargs = {}
    if args.model in PIXEL_BUDGET_MODELS:
        adapter_kwargs["max_pixels"] = args.max_pixels
    adapter = get_adapter(args.model, **adapter_kwargs)
    adapter_dir = Path(args.lora_path).resolve() if args.lora_path else None
    if adapter_dir is not None and not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")
    if adapter_dir is not None and not adapter.supports_lora:
        print(
            f"Warning: --model {adapter.name} does not use LoRA; ignoring adapter path.",
            flush=True,
        )
        adapter_dir = None
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        num_gpus = 0
    device = (
        "cpu"
        if num_gpus == 0
        else (f"cuda:{shard_id % num_gpus}" if num_gpus > 1 else "cuda")
    )
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


def run_cli(args, *, commit_hook: Callable[[], None] | None = None):
    """Orchestration shared by ``main()`` and programmatic callers.

    ``commit_hook`` is invoked after every durable checkpoint write so remote
    storage can snapshot intermediate progress before an interruption; it
    defaults to None for local runs.
    """

    paths = ProjectPaths.from_root(args.project_root)
    args.data_dir = resolve_from_root(args.data_dir, paths.root)
    args.test_json = resolve_from_root(args.test_json, paths.root)
    args.output_dir = resolve_from_root(args.output_dir, paths.root)
    if args.lora_path is not None:
        args.lora_path = resolve_from_root(args.lora_path, paths.root)
    if args.model_path:
        args.model_path = str(resolve_from_root(args.model_path, paths.root))

    if not args.test_json.is_file():
        raise FileNotFoundError(f"Dataset JSON index not found: {args.test_json}")

    adapter_kwargs = {}
    if args.model in PIXEL_BUDGET_MODELS:
        adapter_kwargs["max_pixels"] = args.max_pixels
    adapter = get_adapter(args.model, **adapter_kwargs)

    # Accepts an approved annotation artifact or a flat {query_id: item}
    # index; see the shared inference core for the exact contract.
    items, approved_metadata = load_inference_items(args.test_json, limit=args.limit)

    # ---- Build the run identity -------------------------------------------
    selected_keys = [item["key"] for item in items]
    adapter_dir = Path(args.lora_path).resolve() if args.lora_path else None
    if adapter_dir is not None and not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")
    if adapter_dir is not None and not adapter.supports_lora:
        print(f"Warning: --model {adapter.name} does not use LoRA; ignoring adapter path.")
        adapter_dir = None
    adapter_fingerprint = fingerprint_lora(adapter_dir)
    dataset_for_fp = {item["key"]: item for item in items}
    input_fingerprint = fingerprint_inputs(dataset_for_fp, selected_keys)
    # Bind run identity to the selected samples' image references (paths +
    # recorded sizes when present) without hashing GB of pixels. Existing run
    # dirs keep their recorded fingerprints; only new runs change identity.
    image_fingerprint = trusted_dataset_image_fingerprint(
        dataset_for_fp,
        selected_keys,
        require_recorded_size=False,
    )

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
        image_fingerprint=image_fingerprint,
        num_shards=args.num_shards,
        compute_dtype=getattr(adapter, "compute_dtype", "bfloat16"),
    )
    run_id = metadata["run_id"]
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.json"
    checkpoint_path = run_dir / "checkpoint.json"
    metadata_path = run_dir / "metadata.json"
    print(f"Run id: {run_id}")
    print(f"Output dir: {run_dir}")

    infer_log_file = run_dir / "infer.log"

    def infer_log(msg: str) -> None:
        print(msg, flush=True)
        try:
            with open(infer_log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    predictions = (
        load_resume_predictions(run_dir, metadata, selected_keys)
        if args.resume else {}
    )
    if any(key not in predictions for key in selected_keys) and adapter.name != "mock":
        args.model_path = require_local_model_path(args.model_path)
    # Record identity before any prediction file is published. Existing files
    # have already been checked above; no foreign metadata is silently replaced.
    atomic_write_json(metadata_path, metadata)
    if args.num_shards > 1:
        shard_checkpoint_dir = run_dir / "shard_checkpoints"
        shard_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if predictions:
            atomic_write_json(
                checkpoint_path,
                {"metadata": metadata, "predictions": predictions,
                 "assigned_keys": selected_keys},
            )
            if commit_hook is not None:
                commit_hook()
    if predictions:
        infer_log(f"[{_now_str()}] [EVENT: RUN_RESUME] Resuming inference run {run_id} | Existing: {len(predictions)}/{len(items)} items")
    else:
        infer_log(f"[{_now_str()}] [EVENT: RUN_START] Starting fresh inference run {run_id} | Total items: {len(items)}")
    infer_log(f"Total queries in dataset: {len(items)} | Already finished: {len(predictions)} | Pending: {len(items) - len(predictions)}")
    if args.num_shards > 1:
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
        shard_results = []
        if shards:
            # Executor workers are not daemon processes: their DataLoaders can
            # create CPU workers just like the single-shard path.
            with ProcessPoolExecutor(max_workers=len(shards), mp_context=context) as pool:
                shard_results = list(pool.map(_run_shard_worker, worker_payloads))
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
                commit_hook=commit_hook,
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
                    checkpoint_path=checkpoint_path,
                checkpoint_metadata=metadata,
                commit_hook=commit_hook,
            )
    validate_checkpoint_payload(
        {"metadata": metadata, "predictions": predictions},
        metadata, selected_keys, require_complete=True, label="inference result",
    )
    atomic_write_json(predictions_path, predictions)
    atomic_write_json(
        checkpoint_path,
        {"metadata": metadata, "predictions": predictions,
         "assigned_keys": selected_keys},
    )
    if commit_hook is not None:
        commit_hook()

    infer_log(f"\nInference finished for {len(predictions)} queries. Saved to {predictions_path}")
    infer_log(f"[{_now_str()}] [EVENT: RUN_COMPLETED] Inference completed successfully | Total queries: {len(predictions)}")

    # Metrics only exist when the dataset carries ground-truth bboxes (e.g. val).
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

    atomic_write_json(metadata_path, metadata)
    atomic_write_json(
        checkpoint_path,
        {"metadata": metadata, "predictions": predictions,
         "assigned_keys": selected_keys},
    )
    if commit_hook is not None:
        commit_hook()

    summary = {
        "metadata": metadata,
        "metrics": metrics,
        "total_predictions": len(predictions),
        "valid_predictions": sum(
            validate_bbox(value) is not None for value in predictions.values()
        ),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    print(f"Metadata saved to: {metadata_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Summary saved to: {run_dir / 'summary.json'}")

    return summary


def main():
    run_cli(parse_args())


if __name__ == "__main__":
    main()
