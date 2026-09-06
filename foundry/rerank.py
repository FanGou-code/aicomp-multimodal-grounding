"""Ordinal re-ranking from the enumeration pass (scene-true counts).

The enumeration pass re-counts every cap-saturated frame without the 6-object
cap, twice. Three verdicts per frame (frozen rules):
- count-up (real total > original): ordinals re-ranked over the FULL
  enumeration set — scene-consistent at last;
- count-down (real total < original): the original set contained
  phantom/blurry boxes the cross-pass now rejects; ordinals whose rank exceeds
  the new head count are suppressed;
- inconsistent passes (17%): ordinals suppressed entirely (the teacher cannot
  reliably count this frame).

Also exposes per-head real counts for color arbitration consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from foundry.census import pass_agreement


@dataclass
class FrameVerdict:
    frame_id: str
    verdict: str  # "up" | "down" | "same" | "inconsistent" | "missing"
    real_total: int | None
    original_total: int
    real_counts: dict[str, int]
    real_objects: list[dict]


def load_enumeration(census_dir: Path) -> dict:
    path = Path(census_dir) / "enumeration.json"
    if not path.is_file():
        return {}
    data = json_load(path)
    return data.get("results", {})


def json_load(path: Path) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def original_objects(frame: dict) -> list[dict]:
    if frame.get("single_pass"):
        good = (
            frame["findall_1"]
            if frame["findall_1"]["status"] == "completed"
            else frame["findall_2"]
        )
        return good["objects"]
    return pass_agreement(
        frame["findall_1"]["objects"], frame["findall_2"]["objects"]
    )["agreed_objects"]


def verdict_for(frame_id: str, frame: dict, enum_results: dict) -> FrameVerdict:
    original = original_objects(frame)
    entry = enum_results.get(frame_id)
    if not entry or entry.get("status") != "completed":
        return FrameVerdict(frame_id, "missing", None, len(original), {}, [])
    consistent = entry["n_pass1"] == entry["n_pass2"]
    real_objects = entry.get("objects") or []
    real_total = sum(entry.get("counts", {}).values())
    if not consistent:
        return FrameVerdict(frame_id, "inconsistent", real_total, len(original), entry.get("counts", {}), real_objects)
    if real_total > len(original):
        return FrameVerdict(frame_id, "up", real_total, len(original), entry.get("counts", {}), real_objects)
    if real_total < len(original):
        return FrameVerdict(frame_id, "down", real_total, len(original), entry.get("counts", {}), real_objects)
    return FrameVerdict(frame_id, "same", real_total, len(original), entry.get("counts", {}), real_objects)
