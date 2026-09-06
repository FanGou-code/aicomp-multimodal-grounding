"""Frozen four-bucket classification and quota shares (Phase 0 contract).

Imported by both the assembler and the planner; the classifier wraps the
Phase 0 mining script's frozen regexes so acceptance shares stay
byte-identical with the published test-side counts
(ordinal > distance > spatial > attribute_action).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.phase0_mine_test_style import RE_DIST, RE_ORD, RE_SPAT

FROZEN_BUCKETS = ("ordinal", "distance", "spatial", "attribute_action")
FROZEN_SHARES = {"ordinal": 335, "spatial": 258, "attribute_action": 256, "distance": 151}


def classify_frozen(query: str) -> str:
    """Frozen Phase 0 bucket rule: ordinal > distance > spatial > attribute."""
    if RE_ORD.search(query):
        return "ordinal"
    if RE_DIST.search(query):
        return "distance"
    if RE_SPAT.search(query):
        return "spatial"
    return "attribute_action"


def parse_spec_shares(spec: dict | None) -> dict[str, int]:
    """Per-mille bucket shares from the frozen spec; frozen constants as fallback."""
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
