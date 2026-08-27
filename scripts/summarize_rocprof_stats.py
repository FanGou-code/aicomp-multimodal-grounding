"""Summarize rocprof results.stats.csv by kernel family."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _categorize(name: str) -> str:
    lowered = name.lower()
    if "attention" in lowered or "mha" in lowered or "flash" in lowered:
        return "Attention"
    if "cijk_ailk_bjlk" in lowered or "cijk_ailk_bljk" in lowered:
        return "GEMM"
    if "softmax" in lowered or "logsoftmax" in lowered:
        return "Softmax"
    if "elementwise" in lowered or "vectorized_elementwise" in lowered:
        return "Elementwise"
    if "catarray" in lowered or "index" in lowered or "transpose" in lowered:
        return "DataMovement"
    if "fill" in lowered or "copy" in lowered:
        return "CopyFill"
    if "conv" in lowered or "vision" in lowered:
        return "Vision"
    return "Other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path, default=Path("results.stats.csv"), nargs="?")
    args = parser.parse_args()

    with args.stats.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
        if headers and headers[0].lower() == "name":
            rows = list(reader)
        else:
            rows = []
            if headers is not None:
                rows.append(headers)
            rows.extend(reader)

    totals: dict[str, float] = defaultdict(float)
    groups: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if len(row) < 5:
            continue
        name, count_text, total_text = row[0], row[1], row[2]
        try:
            count = int(count_text)
            total_ns = float(total_text)
        except ValueError:
            continue
        totals[name] += total_ns
        counts[name] += count
        groups[_categorize(name)] += total_ns

    grand_total = sum(totals.values())
    if grand_total <= 0:
        print("No profiled kernels found.")
        return

    print("Top kernels by total duration:")
    for rank, (name, total_ns) in enumerate(sorted(totals.items(), key=lambda item: item[1], reverse=True)[:20], 1):
        print(
            f"{rank:>3} {total_ns / 1e9:9.3f}s "
            f"{total_ns / grand_total * 100:5.1f}% "
            f"{name[:120]}"
        )

    print("\nCategory totals:")
    for group, total_ns in sorted(groups.items(), key=lambda item: item[1], reverse=True):
        print(f"{group:>14} {total_ns / 1e9:9.3f}s {total_ns / grand_total * 100:5.1f}%")


if __name__ == "__main__":
    main()
