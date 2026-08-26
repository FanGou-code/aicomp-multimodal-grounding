"""Analyze the full official test Query set into semantic style groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.query_style import analyze_queries


def load_queries(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an official query mapping, got {type(payload).__name__}")
    queries = [item.get("query") for item in payload.values() if isinstance(item, dict)]
    if any(not isinstance(query, str) or not query for query in queries):
        empty = sum(not isinstance(query, str) or not query for query in queries)
        raise ValueError(f"Official query source contains {empty} empty or invalid query fields")
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/Test/queries/queries.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/annotation_analysis/test_query_templates.json"),
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    analysis = analyze_queries(queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, analysis)
    print(f"Analyzed {analysis['count']} official queries -> {args.output}")
    for style in sorted(analysis["group_ratios"]):
        print(
            f"  {style:<18} {analysis['group_counts'][style]:>5} "
            f"({analysis['group_ratios'][style]:.1%})"
        )


if __name__ == "__main__":
    main()
