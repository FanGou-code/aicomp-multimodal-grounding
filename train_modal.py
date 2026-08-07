"""A100-80GB LoRA training gated by approved self-annotation artifacts."""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path

import modal

from aicomp_grounding.annotation_state import validate_approved_artifact
from aicomp_grounding.artifacts import require_exact_metadata
from aicomp_grounding.bbox import format_qwen_bbox, parse_bbox_from_text
from aicomp_grounding.config import (
    DATA_ROOT,
    MAX_PIXELS,
    MIN_PIXELS,
    MODAL_GPU_PACKAGES,
    MODEL_NAME,
    MODEL_REVISION,
)
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.images import (
    is_trusted_image_fingerprint,
    trusted_dataset_image_fingerprint,
    verify_dataset_images,
)
from aicomp_grounding.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    build_grounding_messages,
    build_training_messages,
    grounding_prompt_hash,
)
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
    validated_prompt_length,
)

dataset_volume = modal.Volume.from_name("rgbdt-dataset")
model_volume = modal.Volume.from_name("hf-model-cache")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(*MODAL_GPU_PACKAGES)
    .add_local_python_source("aicomp_grounding")
)

app = modal.App("rgbdt-visual-grounding", image=image)

OUTPUT_ROOT = Path(DATA_ROOT) / "output_lora"
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
SEED = 42

HYPERPARAMETERS = {
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
    "learning_rate": LEARNING_RATE,
    "epochs": NUM_EPOCHS,
    "warmup_ratio": WARMUP_RATIO,
    "max_grad_norm": MAX_GRAD_NORM,
    "weight_decay": 0.01,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "compute_dtype": "bfloat16",
    "autocast": True,
    "runtime_packages": list(MODAL_GPU_PACKAGES),
    "lora_targets": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
}
GROUNDING_PROMPT_HASH = grounding_prompt_hash(GROUNDING_SYSTEM_PROMPT)


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
            encoded_bbox = format_qwen_bbox(item["bbox"])
            if parse_bbox_from_text(encoded_bbox) is None:
                raise ValueError(
                    f"{split} sample {sample_id!r} collapses after Qwen 0-1000 rounding"
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
    annotation_run_id: str,
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
    root = Path(data_root).resolve()
    annotation_root = root / "outputs" / "annotations" / annotation_run_id
    train_path = annotation_root / "train" / "approved.json"
    val_path = annotation_root / "val" / "approved.json"
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
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        grounding_prompt_hash=GROUNDING_PROMPT_HASH,
        hyperparameters=HYPERPARAMETERS,
        seed=seed,
        run_tag=run_tag,
    )
    run_dir = root / "output_lora" / metadata["training_run_id"]

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
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            num_epochs=NUM_EPOCHS,
        )
        return {
            "metadata": metadata,
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
            batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS, num_epochs=NUM_EPOCHS)
    if latest is not None and not resume:
        raise FileExistsError(
            f"Incomplete training state exists at {latest}; use resume=True or change run_tag"
        )
    return {
        "metadata": metadata,
        "train_artifact_path": str(train_path),
        "val_artifact_path": str(val_path),
        "run_dir": str(run_dir),
        "resume_checkpoint": str(latest) if latest is not None else None,
        "skip_training": False,
        "smoke_test": smoke_test,
        "completed": None,
    }


@app.function(cpu=4.0, memory=8192, timeout=3600, volumes={"/data": dataset_volume})
def preflight_training_environment(
    annotation_run_id: str,
    run_tag: str = "",
    seed: int = SEED,
    resume: bool = True,
    smoke_test: bool = False,
    deep_verify_images: bool = False,
    use_all_data: bool = False,
    val_scenes: int = 40,
) -> dict:
    dataset_volume.reload()
    plan = prepare_training_plan(
        data_root=DATA_ROOT,
        annotation_run_id=annotation_run_id,
        run_tag=run_tag,
        seed=seed,
        resume=resume,
        smoke_test=smoke_test,
        verify_images=deep_verify_images,
        use_all_data=use_all_data,
        val_scenes=val_scenes,
    )
    if plan["resume_checkpoint"]:
        import torch

        checkpoint_path = Path(plan["resume_checkpoint"])
        state = load_json(checkpoint_path / "state.json")
        binary_state = _load_training_state(
            torch, checkpoint_path / "training_state.pt"
        )
        validate_loaded_training_state(binary_state, state, plan["metadata"])
    plan_path = Path(plan["run_dir"]) / "plan.json"
    persisted = {
        "metadata": plan["metadata"],
        "train_artifact_path": plan["train_artifact_path"],
        "val_artifact_path": plan["val_artifact_path"],
    }
    if plan_path.is_file():
        if load_json(plan_path) != persisted:
            raise ValueError(f"Training plan mismatch at {plan_path}")
    elif not smoke_test:
        atomic_write_json(plan_path, persisted)
        dataset_volume.commit()
    return plan


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


@app.function(
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=43200,
    volumes={
        "/data": dataset_volume,
        "/root/.cache/huggingface": model_volume,
    },
)
def train(training_plan: dict) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dataset_volume.reload()
    metadata = training_plan["metadata"]
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
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        grounding_prompt_hash=GROUNDING_PROMPT_HASH,
        hyperparameters=HYPERPARAMETERS,
        seed=metadata["seed"],
        run_tag=metadata["run_tag"],
    )
    require_exact_metadata(metadata, rebuilt_metadata, label="training plan")
    # Modal re-schedules train() on preemption using the original plan
    # (whose resume_checkpoint was None at creation time).  Dynmically
    # re-probe the run directory so a re-scheduled worker picks up any
    # step/epoch checkpoint written before the preemption.
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
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            num_epochs=NUM_EPOCHS,
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

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    class RGBDTGroundingDataset(Dataset):
        def __init__(self, data: dict):
            self.data = data
            self.keys = sorted(data)

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, index):
            item = self.data[self.keys[index]]
            images = []
            for field in ("visible", "infrared", "depth"):
                with Image.open(Path(DATA_ROOT) / item[field]) as opened:
                    images.append(opened.convert("RGB"))
            visible, infrared, depth = images
            bbox_text = format_qwen_bbox(item["bbox"])
            prompt_messages = build_grounding_messages(
                visible, infrared, depth, item["query"]
            )
            messages = build_training_messages(
                visible, infrared, depth, item["query"], bbox_text
            )
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
            prompt_length = validated_prompt_length(
                inputs["input_ids"],
                prompt_inputs["input_ids"],
                sample_id=self.keys[index],
            )
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_length] = -100
            result = {key: value.squeeze(0) for key, value in inputs.items()}
            result["labels"] = labels.squeeze(0)
            return result

    def collate_cpu(batch: list[dict]) -> dict:
        max_length = max(item["input_ids"].size(0) for item in batch)
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Processor tokenizer has no pad_token_id")
        ids, labels, attention, pixels, grids = [], [], [], [], []
        for item in batch:
            padding = max_length - item["input_ids"].size(0)
            ids.append(
                torch.cat(
                    [item["input_ids"], torch.full((padding,), pad_id, dtype=torch.long)]
                )
            )
            labels.append(
                torch.cat(
                    [item["labels"], torch.full((padding,), -100, dtype=torch.long)]
                )
            )
            attention.append(
                torch.cat(
                    [item["attention_mask"], torch.zeros(padding, dtype=torch.long)]
                )
            )
            pixels.append(item["pixel_values"])
            grids.append(item["image_grid_thw"])
        return {
            "input_ids": torch.stack(ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention),
            "pixel_values": torch.cat(pixels, dim=0),
            "image_grid_thw": torch.cat(grids, dim=0),
        }

    train_dataset = RGBDTGroundingDataset(train_artifact["data"])
    val_dataset = RGBDTGroundingDataset(val_artifact["data"])

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        quantization_config=None,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    assert_single_cuda_device_map(getattr(base_model, "hf_device_map", None))
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    if resume_checkpoint:
        model = PeftModel.from_pretrained(
            base_model,
            resume_checkpoint,
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            r=HYPERPARAMETERS["lora_rank"],
            lora_alpha=HYPERPARAMETERS["lora_alpha"],
            lora_dropout=HYPERPARAMETERS["lora_dropout"],
            target_modules=HYPERPARAMETERS["lora_targets"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    def make_train_loader(epoch: int):
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        return DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            collate_fn=collate_cpu,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_cpu,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )
    batches_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    steps_per_epoch = optimizer_steps_per_epoch(batches_per_epoch, GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=HYPERPARAMETERS["weight_decay"],
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch = 0
    resume_batch_index = 0
    global_step = 0
    best_val_loss = float("inf")
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
        best_path = state["best_path"]
        # Restore the running epoch-loss accumulator so the per-epoch loss
        # printout stays correct after a mid-epoch resume.  The step
        # checkpoint stored it; the epoch checkpoint stores the completed
        # epoch average (times a full epoch denominator) which we cannot
        # use as a running total, so default to 0 there.
        resume_epoch_loss = state.get("epoch_loss", 0.0)

    if training_plan.get("smoke_test"):
        smoke_epoch = min(start_epoch, NUM_EPOCHS - 1)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
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

    for epoch in range(start_epoch, NUM_EPOCHS):
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
                GRAD_ACCUM_STEPS,
            )
            (raw_loss / divisor).backward()
            epoch_loss += raw_loss.item()
            if should_optimizer_step(batch_index, len(train_loader), GRAD_ACCUM_STEPS):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 20 == 0:
                    print(
                        f"Epoch {epoch + 1}/{NUM_EPOCHS} | step {global_step}/{total_steps} | "
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
                    dataset_volume.commit()

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

        epoch_adapter_manifest = build_epoch_adapter_manifest(
            metadata,
            epoch=epoch + 1,
            val_loss=average_val_loss,
        )

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_dir = run_dir / "best" / f"epoch_{epoch + 1:02d}"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            atomic_write_json(
                best_dir / "adapter_manifest.json",
                epoch_adapter_manifest,
            )
            best_path = str(best_dir)

        checkpoint_dir = run_dir / "checkpoints" / f"epoch_{epoch + 1:02d}"
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        atomic_write_json(
            checkpoint_dir / "adapter_manifest.json",
            epoch_adapter_manifest,
        )
        checkpoint_state = {
            "metadata": metadata,
            "completed_epoch": epoch + 1,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_path": best_path,
            "train_loss": average_train_loss,
            "val_loss": average_val_loss,
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
        dataset_volume.commit()

    last_dir = run_dir / "last"
    model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)
    atomic_write_json(
        last_dir / "adapter_manifest.json",
        {"metadata": metadata, "completed_epochs": NUM_EPOCHS},
    )
    completed = {
        "metadata": metadata,
        "status": "completed",
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_path": best_path,
        "last_path": str(last_dir),
    }
    validate_completed_training_state(completed, metadata, run_dir,
        batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS, num_epochs=NUM_EPOCHS)
    atomic_write_json(run_dir / "completed.json", completed)
    dataset_volume.commit()
    return completed


@app.local_entrypoint()
def main(
    annotation_run_id: str = "",
    run_tag: str = "",
    seed: int = SEED,
    resume: bool = True,
    preflight_only: bool = False,
    smoke_test: bool = False,
    deep_verify_images: bool = False,
    use_all_data: bool = False,
    val_scenes: int = 40,
) -> dict:
    if not annotation_run_id:
        raise ValueError("annotation_run_id is required; bare train.json/val.json are prohibited")
    plan = preflight_training_environment.remote(
        annotation_run_id=annotation_run_id,
        run_tag=run_tag,
        seed=seed,
        resume=resume,
        smoke_test=smoke_test,
        deep_verify_images=deep_verify_images,
        use_all_data=use_all_data,
        val_scenes=val_scenes,
    )
    if preflight_only:
        print(
            f"Training preflight passed: {plan['metadata']['training_run_id']} | "
            f"already completed: {plan['skip_training']}",
            flush=True,
        )
        return plan

    if plan["skip_training"] and not smoke_test:
        print(f"Training run already completed: {plan['metadata']['training_run_id']}")
        return plan["completed"]

    result = train.remote(plan)
    if smoke_test:
        if result.get("status") != "smoke_passed":
            raise RuntimeError("Training smoke test did not return a passing result")
        print(
            f"Training smoke passed: train_loss={result['train_loss']:.4f}, "
            f"val_loss={result['val_loss']:.4f}"
        )
        return result
    print(f"Training run_id: {result['metadata']['training_run_id']}")
    print(f"Best adapter: {result['best_path']}")
    print(f"Last adapter: {result['last_path']}")
    return result
