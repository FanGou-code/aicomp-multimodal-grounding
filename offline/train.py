"""Offline single-machine LoRA training shell around the shared training core.

Runs the exact shared training pipeline on a local CUDA/HIP GPU. Practical uses:
cheap QLoRA-style experiments on 24GB cards (with a reduced pixel budget),
GroundingDINO-scale fine-tuning, or full runs on >=48GB local hardware.

The portable repository layout keeps approved annotations and training outputs
under repository-level ``outputs/``. ``cloud/train.py`` remains a Modal-only
adapter for the historical volume layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This entrypoint lives in offline/; make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.training_core import SEED, persist_training_plan, prepare_training_plan, run_training
from aicomp_grounding.paths import ProjectPaths, resolve_from_root


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-run-id", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3vl",
        choices=["qwen3vl", "qwen3_8", "internvl35"],
        help="Trainable grounding adapter.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional local base-model directory override (e.g. /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("outputs/annotations"),
        help="Repository-level root containing approved annotation artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Repository-level root for training artifacts (default: outputs).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root; relative data, annotation, and output paths resolve from here.",
    )
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing incomplete checkpoint (default: on).",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--deep-verify-images", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Training DataLoader worker processes. Default 0 avoids forking after "
        "large model/CUDA-HIP context initialization; enable only after a smoke benchmark.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Save/log step checkpoint every N optimizer steps (default 20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.from_root(args.project_root)
    data_root = resolve_from_root(args.data_dir, paths.root)
    annotation_root = resolve_from_root(args.annotation_root, paths.root)
    output_root = resolve_from_root(args.output_root, paths.root)
    plan = prepare_training_plan(
        data_root=data_root,
        annotation_root=annotation_root,
        output_root=output_root,
        annotation_run_id=args.annotation_run_id,
        model=args.model,
        model_path=args.model_path,
        run_tag=args.run_tag,
        seed=args.seed,
        resume=args.resume,
        smoke_test=args.smoke_test,
        verify_images=args.deep_verify_images,
    )
    if not args.smoke_test and not args.preflight_only:
        persist_training_plan(plan)

    if args.preflight_only:
        print(
            "Training preflight passed. Preflight does not persist a run plan; "
            "the authoritative run id is generated and persisted by the GPU/full training run."
        )
        return plan

    if plan["skip_training"] and not args.smoke_test:
        print(f"Training run already completed: {plan['metadata']['training_run_id']}")
        return plan["completed"]

    result = run_training(
        plan,
        data_root=data_root,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
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


if __name__ == "__main__":
    main()
