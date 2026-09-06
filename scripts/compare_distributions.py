#!/usr/bin/env python3
"""Semantic-distribution comparison across annotation versions vs the test set.

Same frozen ruler for every corpus (Phase 0 classifier + spec vocab), so the
v5 assembly can be judged against the two retired annotation versions and the
official test distribution. Read-only, zero API.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.buckets import classify_frozen  # noqa: E402

MAIN_REPO = Path("/home/fang0/dev/projects/aicomp-multimodal-grounding")
PHRASES = [
    "from left to right", "from right to left", "from the left", "from the right",
    "closest to the camera", "farthest from the camera",
    "the far right", "the far left", "of the image", "wearing a",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def load_test() -> list[str]:
    data = json.loads((MAIN_REPO / "data/Test/queries/queries.json").read_text(encoding="utf-8"))
    return [v["query"] for v in data.values()]


def walk_queries(obj, out: list[str]):
    if isinstance(obj, dict):
        q = obj.get("query")
        if isinstance(q, str) and q.strip():
            out.append(q)
        for v in obj.values():
            walk_queries(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_queries(v, out)


def load_golden() -> list[str]:
    data = json.loads((MAIN_REPO / "outputs/annotations/annot_dc189f029d962b27/train/approved.json")
                      .read_text(encoding="utf-8"))
    out: list[str] = []
    walk_queries(data, out)
    return out


def load_old() -> list[str]:
    data = json.loads((MAIN_REPO / "outputs/annotations/annot_ac72f1d926bb2d23/train/merged.json")
                      .read_text(encoding="utf-8"))
    out: list[str] = []
    walk_queries(data, out)
    return out


def load_v5() -> list[str]:
    data = json.loads((PROJECT_ROOT / "outputs/assembly/asm-train-r4/assembly.json")
                      .read_text(encoding="utf-8"))
    return [r["query"] for r in data["records"]]


def stats(texts: list[str], test_vocab5: set[str]) -> dict:
    n = len(texts)
    lower = [t.lower() for t in texts]
    tokens = [tok for t in lower for tok in tokenize(t)]
    buckets = Counter(classify_frozen(t) for t in texts)
    words = [len(t.split()) for t in texts]
    return {
        "n": n,
        "repeat": round(1 - len(set(lower)) / n, 3),
        "words_mean": round(sum(words) / n, 2),
        "words_median": sorted(words)[n // 2],
        "starts_the": round(sum(1 for t in texts if t.startswith("The ")) / n, 3),
        "buckets": {b: round(c / n * 1000) for b, c in buckets.items()},
        "vocab5_coverage": round(sum(1 for tok in tokens if tok in test_vocab5) / max(1, len(tokens)), 3),
        "vocab_size": len(set(tokens)),
        "phrases": {
            p: round(sum(1 for t in lower if p in t) / n * 1000)
            for p in PHRASES
        },
    }


def main() -> None:
    spec = json.loads((PROJECT_ROOT / "spec/vocab_freq.json").read_text(encoding="utf-8"))
    test_vocab5 = {w for w, c in spec.items() if c >= 5}

    corpora = [
        ("test(官方)", load_test()),
        ("旧标注v3", load_old()),
        ("golden v4", load_golden()),
        ("v5组装", load_v5()),
    ]
    computed = {name: stats(texts, test_vocab5) for name, texts in corpora}

    print(f"{'指标':<26}" + "".join(f"{name:>12}" for name, _ in corpora))
    print("-" * (26 + 12 * len(corpora)))

    def row(label, key, fmt=lambda v: v):
        print(f"{label:<26}" + "".join(f"{fmt(computed[name][key]):>12}" for name, _ in corpora))

    row("条数", "n")
    row("逐字重复率", "repeat")
    row("词数均值", "words_mean")
    row("词数中位", "words_median")
    row("The 开头占比", "starts_the", lambda v: f"{v:.0%}")
    row("test高频词覆盖(词例)", "vocab5_coverage", lambda v: f"{v:.0%}")
    row("词表量", "vocab_size")
    print()
    for bucket in ("ordinal", "distance", "spatial", "attribute_action"):
        print(f"{'桶 ' + bucket:<26}" + "".join(
            f"{computed[name]['buckets'].get(bucket, 0):>12}" for name, _ in corpora))
    print("(桶值 = per-mille)")
    print()
    for phrase in PHRASES:
        print(f"{'句型 ' + phrase:<26}" + "".join(
            f"{computed[name]['phrases'][phrase]:>12}" for name, _ in corpora))
    print("(句型值 = per-mille)")


if __name__ == "__main__":
    main()
