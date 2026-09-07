"""Frozen four-bucket classification and quota shares.

Imported by both the assembler and the planner. Classification regexes and
quota shares are derived from self-owned data statistics; the bucket order
is ordinal > distance > spatial > attribute_action.
"""

from __future__ import annotations

import re
from pathlib import Path

RE_ORD = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|last|leftmost|"
    r"rightmost|topmost|bottommost|nearest|closest|farthest)\b",
    re.I,
)
RE_DIST = re.compile(
    r"\b(far|near|close|distance|away|front|behind)\b", re.I,
)
RE_SPAT = re.compile(
    r"\b(left|right|top|bottom|middle|center|centre|corner|beside|below|above|"
    r"under|between|row|edge|side)\b", re.I,
)

FROZEN_BUCKETS = ("ordinal", "distance", "spatial", "attribute_action")
FROZEN_SHARES = {"ordinal": 335, "spatial": 258, "attribute_action": 256, "distance": 151}


def classify_frozen(query: str) -> str:
    """Frozen bucket rule: ordinal > distance > spatial > attribute."""
    if RE_ORD.search(query):
        return "ordinal"
    if RE_DIST.search(query):
        return "distance"
    if RE_SPAT.search(query):
        return "spatial"
    return "attribute_action"


def parse_spec_shares(spec: dict | None) -> dict[str, int]:
    """Per-mille bucket shares; frozen constants as fallback."""
    if spec:
        shares = spec.get("style_buckets_draft", {}).get("shares", {})
        parsed = {}
        for bucket in FROZEN_BUCKETS:
            value = shares.get(bucket)
            if isinstance(value, str) and "per-mille" in value:
                try:
                    parsed[bucket] = int(value.split("(")[1].split("per-mille")[0].strip())
                except (IndexError, ValueError):
                    return dict(FROZEN_SHARES)
            else:
                return dict(FROZEN_SHARES)
        return parsed
    return dict(FROZEN_SHARES)
