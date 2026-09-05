"""Audit the style distribution of generated grounding queries."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.query_style import QUERY_STYLE_GROUPS, analyze_queries

SPATIAL_RE = re.compile(
    r"\b("
    r"left|right|far|near|behind|front|foreground|background|top|bottom|"
    r"middle|center|centre|corner|beside|below|above|under|between|row"
    r")\b",
    re.IGNORECASE,
)
ORDINAL_RE = re.compile(
    r"\b("
    r"first|second|third|fourth|fifth|sixth|seventh|last|"
    r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest"
    r")\b",
    re.IGNORECASE,
)


def extract_queries(payload: object) -> list[str]:
    """Extract query strings from approved/merged/custom query JSON.

    Supported shapes: approved.json (``data -> {sample: {query}}``),
    annotation run results (``results -> {sequence: {frames: {sample:
    {status, query}}}}`` as written to merged.json), or a flat list/dict of
    ``{query}`` items.
    """
    queries: list[str] = []
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, dict):
            for result in results.values():
                if not isinstance(result, dict):
                    continue
                for frame in result.get("frames", {}).values():
                    if (
                        isinstance(frame, dict)
                        and frame.get("status") == "completed"
                        and isinstance(frame.get("query"), str)
                    ):
                        queries.append(frame["query"])
            return queries
        data = payload.get("data", payload)
        if isinstance(data, dict):
            values = list(data.values())
        else:
            values = data if isinstance(data, list) else []
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("query"), str):
                queries.append(value["query"])
    elif isinstance(payload, list):
        for value in payload:
            if isinstance(value, dict) and isinstance(value.get("query"), str):
                queries.append(value["query"])
    return queries


def style_stats(queries: list[str]) -> dict[str, object]:
    """Compute the style distribution of a query list."""
    if not queries:
        raise ValueError("query list is empty")
    word_counts = [len(query.split()) for query in queries]
    spatial = sum(1 for q in queries if SPATIAL_RE.search(q))
    ordinal = sum(1 for q in queries if ORDINAL_RE.search(q))
    either = sum(
        1 for q in queries if SPATIAL_RE.search(q) or ORDINAL_RE.search(q)
    )
    starts_the = sum(1 for q in queries if q.startswith("The "))
    semantic = analyze_queries(queries)
    return {
        "count": len(queries),
        "mean_words": statistics.mean(word_counts),
        "median_words": statistics.median(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "spatial_ratio": spatial / len(queries),
        "ordinal_ratio": ordinal / len(queries),
        "spatial_or_ordinal_ratio": either / len(queries),
        "starts_with_the_ratio": starts_the / len(queries),
        "semantic_group_counts": semantic["group_counts"],
        "semantic_group_ratios": semantic["group_ratios"],
    }


def format_stats_row(label: str, stats: dict[str, object]) -> str:
    return (
        f"{label:<24} n={stats['count']:<6} "
        f"words mean/median={stats['mean_words']:.1f}/{stats['median_words']:.0f} "
        f"(min {stats['min_words']}, max {stats['max_words']})  "
        f"spatial={stats['spatial_ratio']:.1%}  "
        f"ordinal={stats['ordinal_ratio']:.1%}  "
        f"either={stats['spatial_or_ordinal_ratio']:.1%}  "
        f"starts-The={stats['starts_with_the_ratio']:.1%}"
    )


def load_queries(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    queries = extract_queries(payload)
    if not queries:
        raise ValueError(f"no queries found in {path}")
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        nargs="+",
        type=Path,
        required=True,
        help="query JSON files (approved.json, merged.json, or custom query payload)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also print semantic query-group coverage",
    )
    args = parser.parse_args()

    for path in args.queries:
        queries = load_queries(path)
        stats = style_stats(queries)
        print(format_stats_row(path.name, stats))
        if args.full:
            for style in QUERY_STYLE_GROUPS:
                ratio = stats["semantic_group_ratios"][style]
                print(f"  {path.name} {style:<18} {ratio:.1%}")


if __name__ == "__main__":
    main()
