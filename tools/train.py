"""Single-machine LoRA training entrypoint around the shared training core.

Runs the training pipeline on a local CUDA GPU. Practical uses: cheap
QLoRA-style experiments on 24GB cards (with a reduced pixel budget), or full
runs on >=48GB local hardware.

The repository layout keeps approved annotations and training outputs under
repository-level ``outputs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repository root importable when executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.config import load_run_config, merge_run_config
from aicomp_grounding.grounding.engine.training_core import (
    LR_SCHEDULER_TYPES,
    SEED,
    TRAINABLE_MODELS,
    persist_training_plan,
    prepare_training_plan,
    run_training,
)
from aicomp_grounding.paths import ProjectPaths, output_dir, resolve_from_root
from aicomp_grounding.grounding.models.base import require_local_model_path


#: Keys accepted in the ``run:`` section of a --config YAML.
CONFIG_RUN_KEYS = (
    "annotation_run_id",
    "model",
    "model_path",
    "data_dir",
    "annotation_root",
    "output_root",
    "project_root",
    "run_tag",
    "seed",
    "resume",
    "num_workers",
    "checkpoint_interval",
)

#: Keys accepted in the ``hyperparameters:`` section of a --config YAML.
CONFIG_HYPERPARAMETER_KEYS = (
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "epochs",
    "eval_batch_size",
    "best_metric",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "warmup_ratio",
    "lr_scheduler_type",
    "max_grad_norm",
    "weight_decay",
    "max_pixels",
    "gradient_checkpointing",
)

#: Applied after CLI and YAML; ``None`` means "not specified yet".
_RUN_DEFAULTS = {
    "model": "qwen3vl",
    "data_dir": Path("data"),
    "annotation_root": output_dir("annotations"),
    "output_root": Path("outputs"),
    "project_root": Path("."),
    "run_tag": "",
    "seed": SEED,
    "resume": True,
    "num_workers": 0,
    "checkpoint_interval": 20,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML run configuration; CLI flags override it.",
    )
    parser.add_argument("--annotation-run-id", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(TRAINABLE_MODELS),
        help="Trainable grounding adapter (default: qwen3vl).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Pre-downloaded model directory; required for training and smoke tests.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Data root (default: data).")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=None,
        help="Repository-level root containing approved annotation artifacts "
        "(default: outputs/annotations).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Repository-level root for training artifacts (default: outputs).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root; relative data, annotation, and output paths resolve from here.",
    )
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from an existing incomplete checkpoint (default: on).",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Training DataLoader worker processes. Default 0 avoids forking after "
        "large model/CUDA context initialization; enable only after a smoke benchmark.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save/log step checkpoint every N optimizer steps (default 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override adapter default training batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override adapter default gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override adapter default learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override adapter default training epochs.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override adapter default evaluation batch size.",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default=None,
        choices=["acc_at_0_5", "mean_iou", "val_loss"],
        help="Override adapter default best-epoch primary metric.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Override adapter default LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Override adapter default LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="Override adapter default LoRA dropout (0-1).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Override adapter default warmup ratio (0-1).",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default=None,
        choices=list(LR_SCHEDULER_TYPES),
        help="Override adapter default learning-rate schedule.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Override adapter default gradient clipping norm.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override adapter default weight decay.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Override adapter default per-frame pixel budget; must exceed min_pixels "
        "(256*28*28). Lower it if the training GPU is short on VRAM.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override adapter default gradient checkpointing (default: on). "
        "Disable it on GPUs with spare memory to trade memory for speed.",
    )
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
    if not args.annotation_run_id and not args.config:
        parser.error("--annotation-run-id is required when --config is not provided")
    if not args.annotation_run_id:
        args.annotation_run_id = None
    return args


def run_cli(args, *, commit_hook=None):
    """Orchestration shared by ``main()`` and programmatic callers.

    ``commit_hook`` is invoked after every durable write (run plan, training
    checkpoints); it defaults to None because local filesystem writes are
    already atomic.
    """
    paths = ProjectPaths.from_root(args.project_root)
    data_root = resolve_from_root(args.data_dir, paths.root)
    annotation_root = resolve_from_root(args.annotation_root, paths.root)
    output_root = resolve_from_root(args.output_root, paths.root)
    model_path = (
        str(resolve_from_root(args.model_path, paths.root)) if args.model_path else None
    )
    hyperparameter_overrides = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "eval_batch_size": args.eval_batch_size,
        "best_epoch_primary_metric": args.best_metric,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        "max_pixels": args.max_pixels,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    plan = prepare_training_plan(
        data_root=data_root,
        annotation_root=annotation_root,
        output_root=output_root,
        annotation_run_id=args.annotation_run_id,
        model=args.model,
        model_path=model_path,
        run_tag=args.run_tag,
        seed=args.seed,
        resume=args.resume,
        smoke_test=args.smoke_test,
        hyperparameter_overrides=hyperparameter_overrides,
    )
    if args.smoke_test or not plan["skip_training"]:
        plan["model_path"] = require_local_model_path(plan["model_path"])
    if not args.smoke_test:
        persist_training_plan(plan, commit_hook=commit_hook)

    if plan["skip_training"] and not args.smoke_test:
        print(f"Training run already completed: {plan['metadata']['training_run_id']}")
        return plan["completed"]

    result = run_training(
        plan,
        data_root=data_root,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        commit_hook=commit_hook,
    )
    if args.smoke_test:
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


def main():
    run_cli(parse_args())


if __name__ == "__main__":
    main()
