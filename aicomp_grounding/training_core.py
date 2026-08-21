"""Platform-agnostic LoRA training core shared by cloud and offline entrypoints.

Heavy dependencies (torch / peft / transformers) are imported lazily inside
``run_training`` so this module stays importable in torch-free CI environments.

Storage commits after each checkpoint are delegated to an injectable
``commit_hook`` callback: the cloud shell passes its Modal volume commit, the
offline shell passes nothing (local filesystem writes are already atomic).
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path

from aicomp_grounding.artifacts import require_exact_metadata
from aicomp_grounding.bbox import compute_iou, validate_bbox
from aicomp_grounding.config import MODAL_GPU_PACKAGES
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.images import (
    is_trusted_image_fingerprint,
    trusted_dataset_image_fingerprint,
    verify_dataset_images,
)
from aicomp_grounding.models import get_adapter
from aicomp_grounding.models.base import ModelInput
from aicomp_grounding.training_state import (
    accumulation_window_size,
    adapter_weight_path,
    assert_single_cuda_device_map,
    build_epoch_adapter_manifest,
    build_training_metadata,
    expected_global_steps,
    move_batch_to_device,
    optimizer_steps_per_epoch,
    should_optimizer_step,
    validate_adapter_directory,
    validate_completed_training_state,
    validate_loaded_training_state,
    validate_resume_checkpoint,
    validate_training_artifacts,
    validate_epoch_metrics,
    validated_prompt_length,
)

SEED = 42


def _verify_training_images(
    data_root: Path,
    datasets: list[tuple[str, dict]],
) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for split, dataset in datasets:
        fingerprints[split] = verify_dataset_images(
            data_root,
            dataset,
            dataset,
            require_recorded_size=True,
        )
    return fingerprints


def _validate_training_bboxes(datasets: list[tuple[str, dict]]) -> None:
    for split, dataset in datasets:
        for sample_id, item in dataset.items():
            if validate_bbox(item["bbox"]) is None:
                raise ValueError(
                    f"{split} sample {sample_id!r} has an invalid training bbox"
                )


def _evaluate_grounding_metrics(
    *,
    adapter,
    model,
    processor,
    val_data: dict,
    data_root: Path,
    device,
    batch_size: int,
) -> dict:
    """Run full validation-set grounding inference and compute ACC/mIoU."""
    import torch
    from PIL import Image

    keys = sorted(val_data)
    hits = 0
    total_iou = 0.0
    failures = 0
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start : start + batch_size]
        samples = []
        for key in batch_keys:
            item = val_data[key]
            visible = Image.open(data_root / item["visible"]).convert("RGB")
            infrared = Image.open(data_root / item["infrared"]).convert("RGB")
            depth = Image.open(data_root / item["depth"]).convert("RGB")
            samples.append(
                ModelInput(
                    visible=visible,
                    infrared=infrared,
                    depth=depth,
                    query=item["query"],
                    key=key,
                )
            )

        inputs = adapter.build_grounding_batch(samples, processor=processor)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=adapter.generation_config["max_new_tokens"],
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        text_outputs = adapter.decode_grounding_outputs(
            processor,
            generated_ids,
            prompt_len,
        )
        for sample, text in zip(samples, text_outputs):
            item = val_data[sample.key]
            key = sample.key
            ground_truth = validate_bbox(item.get("bbox"))
            if ground_truth is None:
                raise ValueError(f"Validation sample {key!r} has invalid ground-truth bbox")
            prediction = adapter.parse_grounding_text(text)
            if prediction is None:
                failures += 1
                continue
            iou = compute_iou(prediction, ground_truth)
            total_iou += iou
            hits += int(iou >= 0.5)

    total = len(keys)
    return validate_epoch_metrics(
        {
            "hits": hits,
            "total": total,
            "acc_at_0_5": hits / total if total else 0.0,
            "mean_iou": total_iou / total if total else 0.0,
            "failures": failures,
        }
    )


def _latest_resume_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_root = run_dir / "checkpoints"
    candidates = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.iterdir():
            if not path.is_dir():
                continue
            # Accept both epoch_* and step_* as resume sources.  Step
            # checkpoints carry a batch_index so the loader can skip
            # already-trained batches on resume without double-training.
            if not (path.name.startswith("epoch_") or path.name.startswith("step_")):
                continue
            if (
                (path / "state.json").is_file()
                and (path / "training_state.pt").is_file()
                and (path / "adapter_config.json").is_file()
                and adapter_weight_path(path) is not None
            ):
                try:
                    state = load_json(path / "state.json")
                    step = state.get("global_step", 0)
                    candidates.append((step, path))
                except Exception:
                    pass
    return max(candidates, default=(0, None))[1]


def prepare_training_plan(
    *,
    data_root: str | Path,
    annotation_root: str | Path | None = None,
    output_root: str | Path | None = None,
    annotation_run_id: str,
    model: str = "qwen3vl",
    model_path: str | None = None,
    run_tag: str,
    seed: int,
    resume: bool,
    smoke_test: bool = False,
    verify_images: bool = True,
    use_all_data: bool = False,
    val_scenes: int = 40,
) -> dict:
    if not isinstance(smoke_test, bool):
        raise ValueError("smoke_test must be boolean")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", annotation_run_id or ""):
        raise ValueError(f"Invalid annotation_run_id {annotation_run_id!r}")
    adapter_kwargs = {}
    if model == "qwen3vl":
        pass
    elif model == "internvl35":
        adapter_kwargs["max_num_tiles"] = 12
    else:
        raise ValueError(f"Unsupported training model: {model!r}")
    adapter = get_adapter(model, **adapter_kwargs)
    hyperparameters = adapter.training_hyperparameters()
    hyperparameters["runtime_packages"] = list(MODAL_GPU_PACKAGES)
    root = Path(data_root).resolve()
    # Keep the historical Modal layout as the implicit default.  Portable
    # entrypoints pass the repository-level outputs/annotations explicitly so
    # data_root remains responsible only for dataset images and indexes.
    annotation_base = (
        Path(annotation_root).resolve()
        if annotation_root is not None
        else root / "outputs" / "annotations"
    )
    annotation_run_root = annotation_base / annotation_run_id
    train_path = annotation_run_root / "train" / "approved.json"
    val_path = annotation_run_root / "val" / "approved.json"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            "Training requires approved train and val artifacts from the same annotation run: "
            f"{train_path}, {val_path}"
        )
    train_artifact = load_json(train_path)
    val_artifact = load_json(val_path)

    if use_all_data:
        import random as _random_for_split

        merged = {**train_artifact["data"], **val_artifact["data"]}
        scenes: dict[str, list[str]] = {}
        for sample_id in merged:
            scene = sample_id.split("_")[0]
            scenes.setdefault(scene, []).append(sample_id)
        all_scenes = sorted(scenes)
        _random_for_split.Random(seed + 100).shuffle(all_scenes)
        if not 1 <= val_scenes < len(all_scenes):
            raise ValueError(
                f"val_scenes must be in [1, {len(all_scenes) - 1}], got {val_scenes}"
            )
        new_val_scenes = set(all_scenes[:val_scenes])
        train_artifact["data"] = {
            k: v for k, v in merged.items() if k.split("_")[0] not in new_val_scenes
        }
        val_artifact["data"] = {
            k: v for k, v in merged.items() if k.split("_")[0] in new_val_scenes
        }
        # Regenerate dataset and image fingerprints for the new split.
        # dataset_fingerprint: deterministic hash of sample IDs.
        # image_fingerprint: deterministic hash of sample paths (fast; full
        # byte-level verification is deferred to the verify_images path below).
        from aicomp_grounding.artifacts import key_hash, stable_json_hash

        train_artifact["metadata"]["dataset_fingerprint"] = key_hash(
            train_artifact["data"].keys()
        )
        val_artifact["metadata"]["dataset_fingerprint"] = key_hash(
            val_artifact["data"].keys()
        )
        train_image_ids = stable_json_hash(sorted(train_artifact["data"].keys()))
        val_image_ids = stable_json_hash(sorted(val_artifact["data"].keys()))
        train_artifact["metadata"]["image_fingerprint"] = f"key:{train_image_ids}"
        val_artifact["metadata"]["image_fingerprint"] = f"key:{val_image_ids}"
    train_artifact, val_artifact = validate_training_artifacts(
        train_artifact,
        val_artifact,
        annotation_run_id=annotation_run_id,
    )
    datasets = [
        ("train", train_artifact["data"]),
        ("val", val_artifact["data"]),
    ]
    _validate_training_bboxes(datasets)
    for split, artifact in (("train", train_artifact), ("val", val_artifact)):
        recorded = artifact["metadata"]["image_fingerprint"]
        if is_trusted_image_fingerprint(recorded):
            current = trusted_dataset_image_fingerprint(
                artifact["data"],
                artifact["data"],
                require_recorded_size=True,
            )
            if current != recorded:
                raise ValueError(
                    f"{split} image references do not match the approved annotation artifact"
                )
    if verify_images:
        current_image_fingerprints = _verify_training_images(
            root,
            datasets,
        )
        for split, artifact in (("train", train_artifact), ("val", val_artifact)):
            recorded = artifact["metadata"]["image_fingerprint"]
            if (
                not is_trusted_image_fingerprint(recorded)
                and current_image_fingerprints[split] != recorded
            ):
                raise ValueError(
                    f"{split} image bytes do not match the approved annotation artifact"
                )
    metadata = build_training_metadata(
        annotation_run_id=annotation_run_id,
        train_artifact=train_artifact,
        val_artifact=val_artifact,
        model_name=adapter.model_name,
        model_revision=adapter.model_revision,
        grounding_prompt_hash=adapter.prompt_hash(),
        hyperparameters=hyperparameters,
        seed=seed,
        run_tag=run_tag,
    )
    # Keep the historical Modal layout as the implicit default.  Portable
    # entrypoints pass the repository-level outputs directory explicitly so
    # training artifacts do not get mixed into the dataset tree.
    output_base = (
        Path(output_root).resolve()
        if output_root is not None
        else root
    )
    run_dir = output_base / "output_lora" / metadata["training_run_id"]

    if use_all_data:
        run_dir.mkdir(parents=True, exist_ok=True)
        resplit_train = run_dir / "train_resplit.json"
        resplit_val = run_dir / "val_resplit.json"
        atomic_write_json(resplit_train, train_artifact)
        atomic_write_json(resplit_val, val_artifact)
        train_path = resplit_train
        val_path = resplit_val
    completed_path = run_dir / "completed.json"
    if completed_path.is_file():
        completed = validate_completed_training_state(
            load_json(completed_path),
            metadata,
            run_dir,
            batch_size=hyperparameters["batch_size"],
            grad_accum_steps=hyperparameters["gradient_accumulation_steps"],
            num_epochs=hyperparameters["epochs"],
        )
        return {
            "metadata": metadata,
            "model": model,
            "model_path": str(model_path) if model_path is not None else None,
            "train_artifact_path": str(train_path),
            "val_artifact_path": str(val_path),
            "run_dir": str(run_dir),
            "resume_checkpoint": None,
            "skip_training": True,
            "smoke_test": smoke_test,
            "completed": completed,
        }

    latest = _latest_resume_checkpoint(run_dir)
    if latest is not None:
        validate_resume_checkpoint(latest, run_dir=run_dir, metadata=metadata,
            batch_size=hyperparameters["batch_size"],
            grad_accum_steps=hyperparameters["gradient_accumulation_steps"],
            num_epochs=hyperparameters["epochs"])
    if latest is not None and not resume:
        raise FileExistsError(
            f"Incomplete training state exists at {latest}; use resume=True or change run_tag"
        )
    return {
        "metadata": metadata,
        "model": model,
        "model_path": str(model_path) if model_path is not None else None,
        "train_artifact_path": str(train_path),
        "val_artifact_path": str(val_path),
        "run_dir": str(run_dir),
        "resume_checkpoint": str(latest) if latest is not None else None,
        "skip_training": False,
        "smoke_test": smoke_test,
        "completed": None,
    }


def persist_training_plan(plan: dict, *, commit_hook=None) -> None:
    """Persist plan.json for a formal (non-smoke) run; no-op for smoke runs."""
    plan_path = Path(plan["run_dir"]) / "plan.json"
    persisted = {
        "metadata": plan["metadata"],
        "model": plan.get("model", "qwen3vl"),
        "train_artifact_path": plan["train_artifact_path"],
        "val_artifact_path": plan["val_artifact_path"],
    }
    if plan_path.is_file():
        if load_json(plan_path) != persisted:
            raise ValueError(f"Training plan mismatch at {plan_path}")
    elif not plan["smoke_test"]:
        atomic_write_json(plan_path, persisted)
        if commit_hook is not None:
            commit_hook()


def _atomic_torch_save(torch_module, value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch_module.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_training_state(torch_module, path: Path) -> object:
    return torch_module.load(path, map_location="cpu", weights_only=True)


def run_training(
    training_plan: dict,
    *,
    data_root: str | Path,
    commit_hook=None,
) -> dict:
    """Execute the LoRA training run described by ``training_plan``.

    ``data_root`` locates the dataset (images and annotation artifacts);
    ``commit_hook`` is invoked after every checkpoint persistence so remote
    storage (Modal volumes) can snapshot while local runs simply omit it.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader, Dataset

    data_root_path = Path(data_root).resolve()

    def _commit() -> None:
        if commit_hook is not None:
            commit_hook()

    metadata = training_plan["metadata"]
    model_name = training_plan.get("model", "qwen3vl")
    hyperparameters = metadata["hyperparameters"]
    batch_size = hyperparameters["batch_size"]
    grad_accum_steps = hyperparameters["gradient_accumulation_steps"]
    num_epochs = hyperparameters["epochs"]
    learning_rate = hyperparameters["learning_rate"]
    warmup_ratio = hyperparameters["warmup_ratio"]
    max_grad_norm = hyperparameters["max_grad_norm"]
    weight_decay = hyperparameters["weight_decay"]
    eval_batch_size = hyperparameters["eval_batch_size"]
    best_metric_name = hyperparameters["best_epoch_primary_metric"]

    if model_name == "qwen3vl":
        adapter = get_adapter(
            "qwen3vl",
            max_pixels=hyperparameters.get("max_pixels", 3072 * 28 * 28),
        )
    elif model_name == "internvl35":
        adapter = get_adapter(
            "internvl35",
            max_num_tiles=hyperparameters.get("max_num_tiles", 12),
        )
    else:
        raise ValueError(f"Unsupported training model: {model_name!r}")
    run_dir = Path(training_plan["run_dir"])
    train_artifact = load_json(Path(training_plan["train_artifact_path"]))
    val_artifact = load_json(Path(training_plan["val_artifact_path"]))
    validate_training_artifacts(
        train_artifact,
        val_artifact,
        annotation_run_id=metadata["annotation_run_id"],
    )
    rebuilt_metadata = build_training_metadata(
        annotation_run_id=metadata["annotation_run_id"],
        train_artifact=train_artifact,
        val_artifact=val_artifact,
        model_name=adapter.model_name,
        model_revision=adapter.model_revision,
        grounding_prompt_hash=adapter.prompt_hash(),
        hyperparameters=hyperparameters,
        seed=metadata["seed"],
        run_tag=metadata["run_tag"],
    )
    require_exact_metadata(metadata, rebuilt_metadata, label="training plan")
    # Preempted cloud workers are re-scheduled with the original plan (whose
    # resume_checkpoint was None at creation time).  Dynamically re-probe the
    # run directory so a re-scheduled worker picks up any step/epoch
    # checkpoint written before the preemption.
    resume_checkpoint = training_plan.get("resume_checkpoint")
    if not resume_checkpoint:
        latest = _latest_resume_checkpoint(run_dir)
        if latest is not None:
            resume_checkpoint = str(latest)
    if resume_checkpoint:
        validate_resume_checkpoint(
            Path(resume_checkpoint),
            run_dir=run_dir,
            metadata=metadata,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            num_epochs=num_epochs,
        )

    seed = metadata["seed"]
    random_seed = int(seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires the requested CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training configuration requires CUDA bfloat16 support")
    device = torch.device("cuda:0")
    compute_dtype = torch.bfloat16

    model_path = training_plan.get("model_path")
    if model_path is None:
        rel_subpath = (
            "Qwen/Qwen3-VL-8B-Instruct"
            if model_name == "qwen3vl"
            else "OpenGVLab/InternVL3_5-8B-HF"
        )
        for candidate in [
            Path("/mnt/workspace/models") / rel_subpath,
            Path("models") / rel_subpath,
        ]:
            if candidate.is_dir():
                model_path = str(candidate)
                break

    base_model, processor = adapter.load_for_training(device=device, model_path=model_path)

    class RGBDTGroundingDataset(Dataset):
        def __init__(self, data: dict):
            self.data = data
            self.keys = sorted(data)

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, index):
            item = self.data[self.keys[index]]
            return adapter.build_training_batch(
                item,
                data_root=data_root_path,
                processor=processor,
            )

    def collate_cpu(batch: list[dict]) -> dict:
        return adapter.collate_training_batch(batch, processor=processor)

    train_dataset = RGBDTGroundingDataset(train_artifact["data"])
    val_dataset = RGBDTGroundingDataset(val_artifact["data"])

    assert_single_cuda_device_map(getattr(base_model, "hf_device_map", None))
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if resume_checkpoint:
        model = PeftModel.from_pretrained(
            base_model,
            resume_checkpoint,
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            r=hyperparameters["lora_rank"],
            lora_alpha=hyperparameters["lora_alpha"],
            lora_dropout=hyperparameters["lora_dropout"],
            target_modules=adapter.lora_target_modules(),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    def make_train_loader(epoch: int):
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_cpu,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cpu,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )
    batches_per_epoch = math.ceil(len(train_dataset) / batch_size)
    steps_per_epoch = optimizer_steps_per_epoch(batches_per_epoch, grad_accum_steps)
    total_steps = int(steps_per_epoch * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine annealing scheduler with min_lr = 1e-5
    min_lr = learning_rate * 0.1
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr / learning_rate + (1.0 - min_lr / learning_rate) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch = 0
    resume_batch_index = 0
    global_step = 0
    best_val_loss = float("inf")
    best_metric_name = hyperparameters["best_epoch_primary_metric"]
    best_metric_value = -1.0
    best_path = None
    resume_epoch_loss = 0.0
    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        state = load_json(checkpoint_path / "state.json")
        if state.get("metadata") != metadata:
            raise ValueError("Training checkpoint metadata does not match the current plan")
        binary_state = _load_training_state(
            torch, checkpoint_path / "training_state.pt"
        )
        validate_loaded_training_state(binary_state, state, metadata)
        optimizer.load_state_dict(binary_state["optimizer"])
        scheduler.load_state_dict(binary_state["scheduler"])
        torch.set_rng_state(binary_state["torch_rng_state"])
        if torch.cuda.is_available() and binary_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(binary_state["cuda_rng_state"])
        start_epoch = state["completed_epoch"]
        global_step = state["global_step"]
        resume_batch_index = state.get("batch_index", 0)
        best_val_loss = state["best_val_loss"]
        best_metric_name = state.get("best_metric", best_metric_name)
        best_metric_value = state.get("best_metric_value", -1.0)
        best_path = state["best_path"]
        if best_path:
            metrics_path = Path(best_path) / "metrics.json"
            if metrics_path.is_file():
                best_metrics = validate_epoch_metrics(load_json(metrics_path))
                best_metric_name = hyperparameters["best_epoch_primary_metric"]
                best_metric_value = best_metrics[best_metric_name]
        # Restore the running epoch-loss accumulator so the per-epoch loss
        # printout stays correct after a mid-epoch resume.  The step
        # checkpoint stored it; the epoch checkpoint stores the completed
        # epoch average (times a full epoch denominator) which we cannot
        # use as a running total, so default to 0 there.
        resume_epoch_loss = state.get("epoch_loss", 0.0)

    if training_plan.get("smoke_test"):
        smoke_epoch = min(start_epoch, num_epochs - 1)
        smoke_loader = make_train_loader(smoke_epoch)
        try:
            cpu_batch = next(iter(smoke_loader))
            validation_cpu_batch = next(iter(val_loader))
        except StopIteration as exc:
            raise ValueError("Training smoke test requires non-empty train and val loaders") from exc

        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = move_batch_to_device(cpu_batch, device)
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            smoke_train_loss = model(**batch).loss
        if not torch.isfinite(smoke_train_loss):
            raise FloatingPointError("Non-finite training loss in one-batch smoke test")
        smoke_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        model.eval()
        validation_batch = move_batch_to_device(validation_cpu_batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=compute_dtype):
            smoke_val_loss = model(**validation_batch).loss
        if not torch.isfinite(smoke_val_loss):
            raise FloatingPointError("Non-finite validation loss in one-batch smoke test")
        return {
            "metadata": metadata,
            "status": "smoke_passed",
            "train_loss": smoke_train_loss.item(),
            "val_loss": smoke_val_loss.item(),
        }

    for epoch in range(start_epoch, num_epochs):
        train_loader = make_train_loader(epoch)
        # When resuming mid-epoch from a step checkpoint, skip
        # already-trained batches.  The loader uses a deterministic
        # per-epoch seed, so reconstruction produces identical order.
        skip_batches = resume_batch_index if (epoch == start_epoch and resume_batch_index > 0) else 0
        train_iter = iter(train_loader)
        if skip_batches > 0:
            for _ in range(skip_batches):
                next(train_iter)
            print(
                f"Resumed at epoch {epoch + 1}, batch {skip_batches} / {batches_per_epoch}",
                flush=True,
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # On a mid-epoch resume, seed the running loss accumulator with the
        # value stored in the step checkpoint so the loss printout is not
        # distorted by dividing a fresh accumulator by a large batch index.
        epoch_loss = resume_epoch_loss if (epoch == start_epoch and skip_batches > 0) else 0.0
        for batch_index, cpu_batch in enumerate(train_iter, start=skip_batches):
            batch = move_batch_to_device(cpu_batch, device)
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                outputs = model(**batch)
            raw_loss = outputs.loss
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}, batch {batch_index + 1}"
                )
            divisor = accumulation_window_size(
                batch_index,
                len(train_loader),
                grad_accum_steps,
            )
            (raw_loss / divisor).backward()
            epoch_loss += raw_loss.item()
            if should_optimizer_step(batch_index, len(train_loader), grad_accum_steps):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 20 == 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | step {global_step}/{total_steps} | "
                        f"loss {epoch_loss / (batch_index + 1):.4f}",
                        flush=True,
                    )
                    step_dir = run_dir / "checkpoints" / f"step_{global_step:04d}"
                    model.save_pretrained(step_dir)
                    processor.save_pretrained(step_dir)
                    step_state = {
                        "metadata": metadata,
                        "completed_epoch": epoch,
                        "global_step": global_step,
                        "batch_index": batch_index + 1,
                        "best_val_loss": best_val_loss,
                        "best_path": best_path,
                        # Store the RAW running accumulator (not the average)
                        # so a mid-epoch resume can restore it exactly; the
                        # per-step print uses epoch_loss / (batch_index + 1).
                        "epoch_loss": epoch_loss,
                    }
                    atomic_write_json(step_dir / "state.json", step_state)
                    _atomic_torch_save(
                        torch,
                        {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            "completed_epoch": epoch,
                            "global_step": global_step,
                            "training_run_id": metadata["training_run_id"],
                        },
                        step_dir / "training_state.pt",
                    )
                    _commit()

        model.eval()
        validation_loss = 0.0
        validation_batches = 0
        with torch.no_grad():
            for cpu_batch in val_loader:
                batch = move_batch_to_device(cpu_batch, device)
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch + 1}")
                validation_loss += loss.item()
                validation_batches += 1
        average_val_loss = validation_loss / validation_batches
        average_train_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}: train_loss={average_train_loss:.4f}, "
            f"val_loss={average_val_loss:.4f}"
        )

        epoch_metrics = _evaluate_grounding_metrics(
            adapter=adapter,
            model=model,
            processor=processor,
            val_data=val_artifact["data"],
            data_root=data_root_path,
            device=device,
            batch_size=eval_batch_size,
        )
        candidate_metric = epoch_metrics[best_metric_name]
        print(
            f"Epoch {epoch + 1}: ACC@0.5={epoch_metrics['acc_at_0_5']:.4f}, "
            f"mIoU={epoch_metrics['mean_iou']:.4f}, failures={epoch_metrics['failures']}"
        )

        epoch_adapter_manifest = build_epoch_adapter_manifest(
            metadata,
            epoch=epoch + 1,
            val_loss=average_val_loss,
        )

        is_best = (
            candidate_metric > best_metric_value
            or (
                candidate_metric == best_metric_value
                and average_val_loss < best_val_loss
            )
        )
        if is_best:
            best_val_loss = average_val_loss
            best_metric_value = candidate_metric
            best_dir = run_dir / "best" / f"epoch_{epoch + 1:02d}"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            atomic_write_json(
                best_dir / "adapter_manifest.json",
                epoch_adapter_manifest,
            )
            atomic_write_json(best_dir / "metrics.json", epoch_metrics)
            best_path = str(best_dir)

        checkpoint_dir = run_dir / "checkpoints" / f"epoch_{epoch + 1:02d}"
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        atomic_write_json(
            checkpoint_dir / "adapter_manifest.json",
            epoch_adapter_manifest,
        )
        atomic_write_json(checkpoint_dir / "metrics.json", epoch_metrics)
        checkpoint_state = {
            "metadata": metadata,
            "completed_epoch": epoch + 1,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_path": best_path,
            "train_loss": average_train_loss,
            "val_loss": average_val_loss,
            "best_metric": best_metric_name,
            "best_metric_value": best_metric_value,
            "epoch_metrics": epoch_metrics,
        }
        atomic_write_json(checkpoint_dir / "state.json", checkpoint_state)
        _atomic_torch_save(
            torch,
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "completed_epoch": epoch + 1,
                "global_step": global_step,
                "training_run_id": metadata["training_run_id"],
            },
            checkpoint_dir / "training_state.pt",
        )
        _commit()

    last_dir = run_dir / "last"
    model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)
    atomic_write_json(
        last_dir / "adapter_manifest.json",
        {"metadata": metadata, "completed_epochs": num_epochs},
    )
    completed = {
        "metadata": metadata,
        "status": "completed",
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_path": best_path,
        "best_metric": best_metric_name,
        "best_metric_value": best_metric_value,
        "last_path": str(last_dir),
    }
    validate_completed_training_state(completed, metadata, run_dir,
        batch_size=batch_size, grad_accum_steps=grad_accum_steps, num_epochs=num_epochs)
    atomic_write_json(run_dir / "completed.json", completed)
    _commit()
    return completed
