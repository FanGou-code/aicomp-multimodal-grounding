#!/usr/bin/env python3
"""Full corpus audit for assembled query manifests.

Five check groups over an assembly.json (+ its dataset index):
  A. schema & consistency (fields, bbox validity, provenance, edit-log closure)
  B. text layer (grammar residuals, duplicates, non-ascii, casing, echoes)
  C. distribution (frozen four-bucket per-mille vs test)
  D. 11-version lens (skeleton diversity, direction enumeration, extreme,
     nearest/farthest, rank+direction axis)
  E. per-frame uniqueness
Read-only; writes corpus_audit.json next to the manifest.
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

from foundry.buckets import FROZEN_BUCKETS, classify_frozen  # noqa: E402
from scripts.phase0_mine_test_style import tokenize, token_class  # noqa: E402

MAIN_REPO = Path("/home/fang0/dev/projects/aicomp-multimodal-grounding")
REQUIRED_FIELDS = {
    "sample_id", "sequence_id", "source", "category", "bbox", "object_index",
    "query", "family", "bucket", "quota_state", "facts", "words",
}
MISSING_ARTICLE_RE = re.compile(
    r"\b(with|wearing|holding|carrying) (?!(?:a |an |the |its |his |her ))"
)
ECHO_ANCHOR_RE = re.compile(r"closest to the camera")


def load_test() -> list[str]:
    data = json.loads((MAIN_REPO / "data/Test/queries/queries.json").read_text(encoding="utf-8"))
    return [v["query"] for v in data.values()]


def audit(assembly_path: str) -> dict:
    path = Path(assembly_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    records = data.get("records", [])
    split = metadata.get("split", "train")
    index = json.loads((PROJECT_ROOT / "data/indexes" / f"{split}.json").read_text(encoding="utf-8"))

    issues: dict[str, list] = defaultdict(list)
    texts = [r["query"] for r in records]
    lower = [t.lower() for t in texts]

    # ---- A. schema & consistency ----
    for r in records:
        missing = REQUIRED_FIELDS - set(r)
        if missing:
            issues["schema-missing-fields"].append(f"{r.get('sample_id')}: {sorted(missing)}")
        bbox = r.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4
                and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in bbox)
                and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            issues["bbox-invalid"].append(f"{r['sample_id']}#{r['object_index']}: {bbox}")
        if r["source"] not in ("real", "teacher"):
            issues["source-unknown"].append(r["sample_id"])
        if r["sample_id"] not in index:
            issues["sample-not-in-index"].append(r["sample_id"])
        elif r["source"] == "real" and r["bbox"] != index[r["sample_id"]]["bbox"]:
            issues["real-bbox-not-gt"].append(r["sample_id"])
        if r["sequence_id"] != r["sample_id"].split("_", 1)[0]:
            issues["sequence-mismatch"].append(r["sample_id"])

    log_path = path.parent / "text_edits.json"
    edit_log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    edited_records = {f"{r['sample_id']}#{r['object_index']:02d}" for r in records if r.get("edited")}
    log_ids = {e["item_id"] for e in edit_log}
    issues["edit-flag-without-log"] = sorted(edited_records - log_ids)
    issues["edit-log-without-flag"] = sorted(log_ids - edited_records)
    broken_log = [e["item_id"] for e in edit_log if e["before"] == e["after"]]

    # ---- B. text layer ----
    for r in records:
        q = r["query"]
        if not q.strip():
            issues["text-empty"].append(r["sample_id"])
        if q != q.strip() or "  " in q:
            issues["text-whitespace"].append(q)
        if not q[0].isupper():
            issues["text-lowercase-start"].append(q)
        if q[-1] in ".!?":
            issues["text-trailing-punct"].append(q)
        if not q.isascii():
            issues["text-non-ascii"].append(q)
        if re.search(r"\ba a |\ban a \b|_|\bwith (?:in|perched|on) \b", q):
            issues["text-known-bug-pattern"].append(q)
        head = r["category"].split()[-1].lower()
        if head in ("camera", "image"):
            continue  # "closest to the camera" anchor is dialect, not echo
        words = Counter(w for w in tokenize(q) if w == head)
        if words[head] >= 2:
            issues["text-head-echo"].append(q)

    frame_queries: dict[str, list[str]] = defaultdict(list)
    for r in records:
        frame_queries[r["sample_id"]].append(r["query"].lower())
    issues["duplicate-in-frame"] = [
        sid for sid, qs in frame_queries.items() if len(qs) != len(set(qs))
    ]
    global_repeat = round(1 - len(set(lower)) / len(records), 4)
    article_residual = sorted({
        r["query"] for r in records if MISSING_ARTICLE_RE.search(r["query"])
    })

    # ---- C. distribution ----
    buckets = Counter(classify_frozen(t) for t in texts)
    bucket_per_mille = {b: round(buckets[b] / len(records) * 1000) for b in FROZEN_BUCKETS}
    words = [r["words"] for r in records]

    # ---- D. 11-version lens ----
    skeletons = Counter(" ".join(token_class(x) for x in tokenize(t)) for t in texts)
    spec = json.loads((PROJECT_ROOT / "spec/style_spec.json").read_text(encoding="utf-8"))
    top60 = {s["skeleton"] for s in spec["skeletons_top60"]}

    def per_mille(pattern: str) -> int:
        return round(sum(1 for t in lower if re.search(pattern, t)) / len(records) * 1000)

    lens = {
        "unique_skeletons": len(skeletons),
        "top60_coverage": round(sum(c for s, c in skeletons.items() if s in top60) / len(records), 3),
        "direction_enumeration": per_mille(r"from (left|right) to (left|right)"),
        "extreme_most": per_mille(r"\b(leftmost|rightmost|topmost|bottommost)\b"),
        "nearest_farthest": per_mille(r"\b(nearest|farthest)\b"),
        "rank_direction_axis": per_mille(
            r"\b(first|second|third|fourth|fifth|last)\b.*\b(left|right)\b"),
    }

    return {
        "manifest": metadata.get("run_tag", path.parent.name),
        "count": len(records),
        "sources": dict(Counter(r["source"] for r in records)),
        "edited": len(edited_records),
        "verbatim_repeat": global_repeat,
        "words_mean": round(sum(words) / len(records), 2),
        "starts_the": round(sum(1 for t in texts if t.startswith("The ")) / len(records), 3),
        "bucket_per_mille": bucket_per_mille,
        "lens": lens,
        "article_residual_kept": len(article_residual),
        "issues": {k: v for k, v in issues.items() if v},
        "issue_counts": {k: len(v) for k, v in issues.items() if v},
    }


def main() -> None:
    results = {}
    for arg in sys.argv[1:]:
        results[Path(arg).parent.name] = audit(arg)
        (Path(arg).parent / "corpus_audit.json").write_text(
            json.dumps(results[Path(arg).parent.name], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    test_texts = load_test()
    test_buckets = Counter(classify_frozen(t) for t in test_texts)
    test_pm = {b: round(c / len(test_texts) * 1000) for b, c in test_buckets.items()}

    print(f"{'审计组':<30}" + "".join(f"{name:>16}" for name in results))
    print("-" * (30 + 16 * len(results)))
    rows = [
        ("条数", lambda r: r["count"]),
        ("real / teacher", lambda r: f"{r['sources'].get('real',0)}/{r['sources'].get('teacher',0)}"),
        ("人审修改记录", lambda r: r["edited"]),
        ("逐字重复", lambda r: f"{r['verbatim_repeat']:.4f}"),
        ("词数均值", lambda r: r["words_mean"]),
        ("The 开头", lambda r: f"{r['starts_the']:.0%}"),
    ]
    for label, fn in rows:
        print(f"{label:<30}" + "".join(f"{str(fn(r)):>16}" for r in results.values()))
    print(f"{'桶 per-mille (test 参照)':<30}"
          + "".join(f"{'':>16}" for _ in results))
    for bucket in FROZEN_BUCKETS:
        print(f"  {bucket:<28}" + "".join(
            f"{r['bucket_per_mille'].get(bucket, 0):>16}" for r in results.values()))
    print(f"  {'(test 参照)':<27}"
          + "".join(f"{test_pm.get(b, 0):>16}" for b in FROZEN_BUCKETS for _ in [0][:1])[:16] * len(results)
          if False else
          f"  {'(test 参照)':<27}" + f"{test_pm['ordinal']:>10}/{test_pm['distance']} / {test_pm['spatial']} / {test_pm['attribute_action']}")
    print(f"{'11版尺子':<30}" + "".join(f"{'':>16}" for _ in results))
    for key in ("unique_skeletons", "top60_coverage", "direction_enumeration",
                "extreme_most", "nearest_farthest", "rank_direction_axis"):
        print(f"  {key:<28}" + "".join(f"{str(r['lens'][key]):>16}" for r in results.values()))
    print(f"{'问题计数':<30}" + "".join(f"{'':>16}" for _ in results))
    all_keys = sorted({k for r in results.values() for k in r["issue_counts"]})
    for key in all_keys:
        print(f"  {key:<28}" + "".join(f"{r['issue_counts'].get(key, 0):>16}" for r in results.values()))
    for name, r in results.items():
        print(f"\n[{name}] 冠词保留(裁决放行): {r['article_residual_kept']}")
        for k, v in r["issues"].items():
            print(f"  {k}: {v[:3]}{' ...' if len(v) > 3 else ''}")


from collections import defaultdict  # noqa: E402

if __name__ == "__main__":
    main()
