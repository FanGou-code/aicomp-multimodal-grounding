"""CLI tool to register official test set scores and bind them to run_id provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.leaderboard import (
    DEFAULT_REGISTRY_PATH,
    format_leaderboard_table,
    register_score,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register official competition submission scores to the project provenance tree."
    )
    parser.add_argument("--score", type=float, help="Official competition test score (e.g. 0.6471)")
    parser.add_argument("--tag", default="", help="Label for this run (e.g. Base, QLoRA-v1)")
    parser.add_argument("--notes", default="", help="Brief description or observations")
    parser.add_argument("--submission-zip", type=Path, help="Path to submission.zip")
    parser.add_argument("--inference-run-id", default="", help="Inference run_id")
    parser.add_argument("--adapter-run-id", default="", help="Training adapter run_id")
    parser.add_argument("--annotation-run-id", default="", help="Annotation run_id")
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--show", action="store_true", help="Display the registered leaderboard table")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.show or args.score is None:
        print("\n=== Leaderboard Score Registry ===\n")
        print(format_leaderboard_table(args.registry_path))
        print()
        if args.score is None:
            return

    record = register_score(
        official_score=args.score,
        tag=args.tag,
        notes=args.notes,
        submission_zip=args.submission_zip,
        inference_run_id=args.inference_run_id,
        adapter_run_id=args.adapter_run_id,
        annotation_run_id=args.annotation_run_id,
        registry_path=args.registry_path,
    )
    print(f"Successfully registered score {record['official_score']:.4f} (Tag: {record['tag']})")
    print(format_leaderboard_table(args.registry_path))


if __name__ == "__main__":
    main()
