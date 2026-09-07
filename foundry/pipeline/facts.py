"""Shared fact-layer data types and frame-fact extraction (pure data, no policy).

``ObjectFacts`` is one enumerated object's derived facts; ``Realization`` is
one candidate sentence. Both are consumed by the assembler and the planner.
``extract_frame_facts`` turns one frame's trusted enumeration + attr card +
depth record into ``ObjectFacts``; all geometric margins live here so the
extraction thresholds have exactly one home.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from foundry.bbox import compute_iou
from foundry.pipeline.depth import (
    DEPTH_SOURCE,
    is_background as depth_in_background,
    is_farthest as depth_farthest,
    is_foreground as depth_in_foreground,
    is_nearest as depth_nearest,
)

CANARY_IOU = 0.5
ORDINAL_GAP = 0.02
EXTREME_MARGIN = 0.04
DIST_MARGIN = 0.04
ANCHOR_GAP = 0.01
SIDE_OF_IMAGE_THRESHOLDS = (1.0 / 3.0, 2.0 / 3.0)
MAX_ANCHORS = 3
COLOR_WORD_RE = re.compile(r"^[a-z]+$")

def category_head(category: str) -> str:
    words = category.split()
    return words[-1] if words else category


def article_for(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


@dataclass(frozen=True)
class ObjectFacts:
    index: int
    category: str
    bbox: tuple[float, float, float, float]
    color: str | None
    features: str | None
    is_canary: bool
    count_in_head: int
    rank_left: int | None
    rank_right: int | None
    is_leftmost: bool
    is_rightmost: bool
    is_topmost: bool
    is_bottommost: bool
    is_closest: bool
    is_farthest: bool
    side_of_image: str | None
    anchors_left: tuple[tuple[int, str], ...]
    anchors_right: tuple[tuple[int, str], ...]
    median_mm: int | None = None
    is_in_foreground: bool = False
    is_in_background: bool = False
    area_ratio_lead: float | None = None

    @property
    def head(self) -> str:
        return category_head(self.category)

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class Realization:
    text: str
    family: str
    facts: tuple[str, ...]
    words: int


def _clean_color(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    word = raw.strip().lower().split()[0]
    if COLOR_WORD_RE.fullmatch(word) and word not in ("a", "an", "the"):
        return word
    return None


def _clean_features(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw.strip().rstrip(".").lower())
    if not text or not text.isascii() or len(text) > 32:
        return None
    return text


def _strip_head_prefix(features: str, head: str) -> str:
    """Drop a redundant "{head} (with )" prefix echoed by the teacher.

    Attr features sometimes restate the category ("swan with neck curved
    downward"); keeping it would realize "the swan with swan with ...".
    """
    text = features
    for _ in range(2):
        if text == head:
            return ""
        if text.startswith(head + " with "):
            text = text[len(head) + 6:]
        elif text.startswith(head + " "):
            text = text[len(head) + 1:]
        elif text.startswith("with "):
            text = text[5:]
        else:
            break
    return text


def _exclusive(value: float, others: list[float], margin: float, *, higher: bool) -> bool:
    """True when value leads every other entry by at least ``margin``."""
    if not others:
        return False
    if higher:
        return value >= max(others) + margin
    return value <= min(others) - margin


def extract_frame_facts(
    objects: list[dict],
    gt_bbox: list[float],
    attr: dict | None,
    depth_facts: dict | None = None,
) -> list[ObjectFacts]:
    """Derive per-object facts from one frame's trusted enumeration.

    ``depth_facts`` is the census frame's stored depth record; when present,
    closest/farthest switch from the y2 geometric proxy to exact millimeter
    margins and the foreground/background band facts become available.
    """
    depth_medians: dict[int, int | None] = {}
    frame_min_mm = frame_max_mm = None
    if depth_facts and depth_facts.get("source") == DEPTH_SOURCE:
        frame_min_mm = depth_facts.get("frame_min_mm")
        frame_max_mm = depth_facts.get("frame_max_mm")
        depth_medians = {
            int(index): entry.get("median_mm")
            for index, entry in (depth_facts.get("objects") or {}).items()
        }
    cleaned: list[dict] = []
    for obj in objects:
        bbox = obj["bbox"]
        attr_entry = (attr or {}).get(str(obj["i"])) or (attr or {}).get(obj["i"]) or {}
        category = " ".join(str(obj["category"]).strip().lower().split())
        features = _clean_features(attr_entry.get("features"))
        if features:
            features = _strip_head_prefix(features, category_head(category)) or None
        cleaned.append(
            {
                "index": obj["i"],
                "category": category,
                "bbox": bbox,
                "color": _clean_color(attr_entry.get("color")),
                "features": features,
                "is_canary": compute_iou(bbox, gt_bbox) >= CANARY_IOU,
            }
        )

    centers_x = [(c["bbox"][0] + c["bbox"][2]) / 2 for c in cleaned]
    centers_y = [(c["bbox"][1] + c["bbox"][3]) / 2 for c in cleaned]
    bottoms = [c["bbox"][3] for c in cleaned]
    head_counts: dict[str, int] = {}
    for c in cleaned:
        head = category_head(c["category"])
        head_counts[head] = head_counts.get(head, 0) + 1
    by_head: dict[str, list[int]] = {}
    for pos, c in enumerate(cleaned):
        by_head.setdefault(category_head(c["category"]), []).append(pos)

    lo_x, hi_x = SIDE_OF_IMAGE_THRESHOLDS
    facts: list[ObjectFacts] = []
    for pos, c in enumerate(cleaned):
        head = category_head(c["category"])
        group = by_head[head]
        rank_left = rank_right = None
        if len(group) >= 2:
            ordered = sorted(group, key=lambda p: centers_x[p])
            rank = ordered.index(pos) + 1
            gap_ok = True
            for neighbour in (rank - 2, rank):
                if 0 <= neighbour < len(ordered):
                    if abs(centers_x[ordered[neighbour]] - centers_x[pos]) < ORDINAL_GAP:
                        gap_ok = False
                        break
            if gap_ok:
                rank_left = rank
                rank_right = len(group) + 1 - rank

        def anchors(phrase_side: str) -> tuple[tuple[int, str], ...]:
            """Anchor candidates for "on the <side> side of the <anchor>".

            phrase_side "left" means the target sits on the anchor's left, so
            the anchor lies wholly to the target's right (and vice versa).
            Anchors with a non-unique head noun are skipped: an ambiguous
            anchor cannot name the relation.
            """
            candidates = []
            for q in range(len(cleaned)):
                if q == pos:
                    continue
                anchor_head = category_head(cleaned[q]["category"])
                if head_counts[anchor_head] != 1:
                    continue
                if phrase_side == "left":
                    if cleaned[q]["bbox"][0] - c["bbox"][2] < ANCHOR_GAP:
                        continue
                    gap = centers_x[q] - centers_x[pos]
                else:
                    if c["bbox"][0] - cleaned[q]["bbox"][2] < ANCHOR_GAP:
                        continue
                    gap = centers_x[pos] - centers_x[q]
                candidates.append((gap, q))
            candidates.sort()
            return tuple(
                (cleaned[q]["index"], cleaned[q]["category"]) for _, q in candidates[:MAX_ANCHORS]
            )

        median_mm = depth_medians.get(c["index"])
        # Area-comparative fact: lead ratio over the same-head runner-up
        # (admin rule 2026-09-06: derivable from bboxes, no absolute size
        # thresholds). None when no same-head rival exists.
        area = max(1e-9, (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]))
        rival_areas = [
            max(1e-9, (cleaned[q]["bbox"][2] - cleaned[q]["bbox"][0]) * (cleaned[q]["bbox"][3] - cleaned[q]["bbox"][1]))
            for q in group if q != pos
        ]
        area_ratio_lead = round(area / max(rival_areas), 3) if rival_areas else None
        other_medians = [
            m
            for q in range(len(cleaned))
            if q != pos and (m := depth_medians.get(cleaned[q]["index"])) is not None
        ]
        if median_mm is not None:
            is_closest = depth_nearest(median_mm, other_medians)
            is_farthest = depth_farthest(median_mm, other_medians)
        else:
            is_closest = _exclusive(
                bottoms[pos], bottoms[:pos] + bottoms[pos + 1:], DIST_MARGIN, higher=True
            )
            is_farthest = _exclusive(
                bottoms[pos], bottoms[:pos] + bottoms[pos + 1:], DIST_MARGIN, higher=False
            )
        facts.append(
            ObjectFacts(
                index=c["index"],
                category=c["category"],
                bbox=tuple(c["bbox"]),
                color=c["color"],
                features=c["features"],
                is_canary=c["is_canary"],
                count_in_head=head_counts[head],
                rank_left=rank_left,
                rank_right=rank_right,
                is_leftmost=_exclusive(
                    centers_x[pos], centers_x[:pos] + centers_x[pos + 1:], EXTREME_MARGIN, higher=False
                ),
                is_rightmost=_exclusive(
                    centers_x[pos], centers_x[:pos] + centers_x[pos + 1:], EXTREME_MARGIN, higher=True
                ),
                is_topmost=_exclusive(
                    centers_y[pos], centers_y[:pos] + centers_y[pos + 1:], EXTREME_MARGIN, higher=False
                ),
                is_bottommost=_exclusive(
                    centers_y[pos], centers_y[:pos] + centers_y[pos + 1:], EXTREME_MARGIN, higher=True
                ),
                is_closest=is_closest,
                is_farthest=is_farthest,
                side_of_image=(
                    "left" if centers_x[pos] < lo_x else "right" if centers_x[pos] > hi_x else None
                ),
                anchors_left=anchors("left"),
                anchors_right=anchors("right"),
                median_mm=median_mm,
                is_in_foreground=depth_in_foreground(median_mm, frame_min_mm, frame_max_mm),
                is_in_background=depth_in_background(median_mm, frame_min_mm, frame_max_mm),
                area_ratio_lead=area_ratio_lead,
            )
        )
    return facts
