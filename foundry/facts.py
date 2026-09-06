"""Shared fact-layer data types (pure data, no policy).

``ObjectFacts`` is one enumerated object's derived facts; ``Realization`` is
one candidate sentence. Both are consumed by the assembler and the planner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from foundry.bbox import compute_iou

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
