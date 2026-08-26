"""Materialize a deterministic expanded annotation source from scene cards.

The generated directory is a compatible `generate_queries.py` data root:
it contains `<split>.json`, `split_manifest.json`, and relative image symlinks
back to the original `data/raw/Train` and `data/derived/Processed` trees. Raw
indexes are never modified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.config import (
    ANNOTATION_SPLITS,
    PREPARATION_PROTOCOL_VERSION,
)
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.query_style import (
    build_style_plan,
    expand_annotation_source,
    style_plan_fingerprint,
)


def _ensure_image_links(expanded_root: Path, data_root: Path) -> None:
    expanded_root.mkdir(parents=True, exist_ok=True)
    link_specs = (
        ("Train", data_root / "raw" / "Train"),
        ("Processed", data_root / "derived" / "Processed"),
    )
    for name, target in link_specs:
        link = expanded_root / name
        if link.exists() or link.is_symlink():
            continue
        target = target.resolve()
        if not target.is_dir():
            raise FileNotFoundError(
                f"Cannot create expanded image link, missing {target}"
            )
        os.symlink(os.path.relpath(target, expanded_root), link)


def build_style_plan_artifacts(
    *,
    split: str,
    data_root: Path,
    index_root: Path,
    scene_cards_path: Path,
    output_root: Path,
    seed: int = 42,
    queries_per_frame: int = 3,
) -> dict:
    split = split.strip().lower()
    if split not in ANNOTATION_SPLITS:
        raise ValueError(f"Annotation is restricted to train/val, got {split!r}")
    raw = load_json(index_root / f"{split}.json")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{split} annotation source is empty")
    cards = load_json(scene_cards_path)
    if not isinstance(cards, dict):
        raise ValueError(f"Scene cards must be an object: {scene_cards_path}")

    plan = build_style_plan(
        raw,
        cards,
        seed=seed,
        queries_per_frame=queries_per_frame,
    )
    expanded = expand_annotation_source(raw, plan)

    expanded_root = output_root.resolve() / f"expanded_{split}"
    if expanded_root.is_dir():
        for old in ("style_plan.json", f"{split}.json", "split_manifest.json"):
            (expanded_root / old).unlink(missing_ok=True)
    _ensure_image_links(expanded_root, data_root.resolve())

    manifest = load_json(index_root / "split_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError("Raw split_manifest.json is not a completed supported preparation")
    manifest["preparation_protocol_version"] = PREPARATION_PROTOCOL_VERSION
    manifest["index_fingerprints"] = dict(manifest.get("index_fingerprints", {}))
    manifest["index_sample_counts"] = dict(manifest.get("index_sample_counts", {}))
    manifest["index_fingerprints"][split] = stable_json_hash(expanded)
    manifest["index_sample_counts"][split] = len(expanded)
    if split == "train":
        manifest["stats"] = dict(manifest.get("stats", {}))
        manifest["stats"]["train_samples"] = len(expanded)
    else:
        manifest["stats"] = dict(manifest.get("stats", {}))
        manifest["stats"]["val_samples"] = len(expanded)
    manifest["expanded_style_plan_fingerprint"] = style_plan_fingerprint(plan)
    manifest["queries_per_frame"] = queries_per_frame

    atomic_write_json(expanded_root / f"{split}.json", expanded)
    atomic_write_json(expanded_root / "split_manifest.json", manifest)
    atomic_write_json(expanded_root / "style_plan.json", plan)
    return {
        "split": split,
        "raw_samples": len(raw),
        "expanded_samples": len(expanded),
        "scene_sequences": len(cards),
        "style_plan_path": str(expanded_root / "style_plan.json"),
        "expanded_source_path": str(expanded_root / f"{split}.json"),
        "data_root_for_generation": str(expanded_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(ANNOTATION_SPLITS), default="train")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--index-root", type=Path, default=Path("data/indexes"))
    parser.add_argument("--scene-cards", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/annotation_analysis"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries-per-frame", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.queries_per_frame <= 6:
        parser.error("--queries-per-frame must be between 1 and 6")
    result = build_style_plan_artifacts(
        split=args.split,
        data_root=args.data_root,
        index_root=args.index_root,
        scene_cards_path=args.scene_cards,
        output_root=args.output_root,
        seed=args.seed,
        queries_per_frame=args.queries_per_frame,
    )
    print(f"raw_samples={result['raw_samples']} expanded_samples={result['expanded_samples']}")
    print(f"style_plan: {result['style_plan_path']}")
    print(f"generation data root: {result['data_root_for_generation']}")


if __name__ == "__main__":
    main()
