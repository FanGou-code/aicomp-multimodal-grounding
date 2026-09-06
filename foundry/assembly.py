"""Local query assembly from census facts (三权分立: 代码只说).

Every emitted sentence is (a) supported by the census facts of its own frame
and (b) verified unique inside that frame by a deterministic matcher — the
description must resolve to exactly one enumerated object, so assembled
queries cannot be ambiguous supervision. The teacher never writes query text.

Disambiguation groups are keyed on the category HEAD noun (last word), the
noun a query text can actually name: "swan" and "black swan" share the head
"swan" and are therefore one group for ordinal ranks and uniqueness checks.
Geometric facts use strict margins so each extreme holds for at most one
object per frame:
- ordinal: category has >= 2 instances ordered by x1 and the object's
  center-x gap to both rank neighbours is >= ORDINAL_GAP
- leftmost/rightmost (center-x), topmost/bottommost (center-y),
  closest/farthest (bottom edge y2, ground-view proximity proxy), each
  leading every other object by >= the respective margin
- side of image (center-x thirds) and side of a unique-head anchor object
  (wholly left/right of it with clear separation)
- attributes: per-object color and feature strings from the census attr pass

The bucket classifier is imported from the frozen Phase 0 mining script so
acceptance shares stay byte-identical with the published test-side counts
(ordinal > distance > spatial > attribute_action; superlatives fall in the
ordinal bucket under this frozen rule).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace

from foundry.bbox import compute_iou
from foundry.census import pass_agreement
from foundry.rerank import verdict_for
from foundry.facts import Realization, article_for, category_head  # noqa: F401
from foundry.facts import ObjectFacts  # noqa: F401
from foundry.planner import TargetSupply, plan as planner_plan
from foundry.depth import (
    DEPTH_SOURCE,
    is_background as depth_in_background,
    is_farthest as depth_farthest,
    is_foreground as depth_in_foreground,
    is_nearest as depth_nearest,
)
from foundry.buckets import (
    FROZEN_BUCKETS,
    FROZEN_SHARES,
    classify_frozen,
    parse_spec_shares,
)

ORDINAL_WORDS = [
    "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth",
]
ACTION_FEATURE_RE = re.compile(r"^[a-z]+ing\b")
COLOR_WORD_RE = re.compile(r"^[a-z]+$")
CANARY_IOU = 0.5

# Query texts never combine bare color adjectives with person heads ("the
# white person" reads as a race descriptor): test dialect colors people via
# "wearing ..." feature phrases instead.
COLOR_SUPPRESSED_HEADS = frozenset(
    {"person", "child", "man", "woman", "men", "women", "people", "couple"}
)

ORDINAL_GAP = 0.02
EXTREME_MARGIN = 0.04
DIST_MARGIN = 0.04
ANCHOR_GAP = 0.01
SIDE_OF_IMAGE_THRESHOLDS = (1.0 / 3.0, 2.0 / 3.0)
MIN_TEACHER_AREA = 0.002
MAX_TEACHER_AREA = 0.6
MAX_ANCHORS = 3




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


def _variant(text: str, family: str, facts: tuple[str, ...]) -> Realization:
    return Realization(" ".join(text.split()), family, facts, len(text.split()))


def _color_slot(f: ObjectFacts) -> str:
    if f.color and f.head not in COLOR_SUPPRESSED_HEADS:
        return f"{f.color} "
    return ""


def _ordinal_word(rank: int) -> str | None:
    return ORDINAL_WORDS[rank - 1] if 1 <= rank <= len(ORDINAL_WORDS) else None


def _realize_ordinal(f: ObjectFacts) -> list[Realization]:
    out: list[Realization] = []
    color_slot = _color_slot(f)
    for rank, directions in (
        (f.rank_left, ("from the left", "from left to right")),
        (f.rank_right, ("from the right", "from right to left")),
    ):
        word = _ordinal_word(rank) if rank else None
        if word is None:
            continue
        for direction in directions:
            out.append(_variant(
                f"The {word} {color_slot}{f.head} {direction}",
                "ordinal_direction", (f"rank:{rank}", direction),
            ))
    return out


def _realize_superlative(f: ObjectFacts) -> list[Realization]:
    out: list[Realization] = []
    color_slot = _color_slot(f)
    if f.is_closest:
        out.append(_variant(f"The {color_slot}{f.head} closest to the camera", "superlative_camera", ("y2-max",)))
        out.append(_variant(f"The closest {color_slot}{f.head}", "superlative_camera", ("y2-max",)))
    if f.is_farthest:
        out.append(_variant(f"The {color_slot}{f.head} farthest from the camera", "superlative_camera", ("y2-min",)))
        out.append(_variant(f"The farthest {color_slot}{f.head}", "superlative_camera", ("y2-min",)))
    if f.is_leftmost:
        out.append(_variant(f"The {color_slot}{f.head} on the far left", "superlative_camera", ("x-min",)))
        out.append(_variant(f"The leftmost {color_slot}{f.head}", "superlative_camera", ("x-min",)))
    if f.is_rightmost:
        out.append(_variant(f"The {color_slot}{f.head} on the far right", "superlative_camera", ("x-max",)))
        out.append(_variant(f"The rightmost {color_slot}{f.head}", "superlative_camera", ("x-max",)))
    if f.is_topmost:
        out.append(_variant(f"The topmost {color_slot}{f.head}", "superlative_camera", ("y-min",)))
    if f.is_bottommost:
        out.append(_variant(f"The bottommost {color_slot}{f.head}", "superlative_camera", ("y-max",)))
    if f.is_in_foreground:
        out.append(_variant(f"The {color_slot}{f.head} in the foreground".replace("  ", " "), "superlative_camera", ("foreground",)))
    if f.is_in_background:
        out.append(_variant(f"The {color_slot}{f.head} in the background".replace("  ", " "), "superlative_camera", ("background",)))
    return out


def _realize_anchor(f: ObjectFacts) -> list[Realization]:
    out: list[Realization] = []
    color_slot = _color_slot(f)
    if f.side_of_image:
        out.append(_variant(
            f"The {color_slot}{f.head} on the {f.side_of_image} side of the image",
            "side_of_anchor", (f"image:{f.side_of_image}",),
        ))
    for anchor_index, anchor_category in f.anchors_left:
        out.append(_variant(
            f"The {color_slot}{f.head} on the left side of the {category_head(anchor_category)}",
            "side_of_anchor", (f"anchor-left:{anchor_index}:{anchor_category}",),
        ))
    for anchor_index, anchor_category in f.anchors_right:
        out.append(_variant(
            f"The {color_slot}{f.head} on the right side of the {category_head(anchor_category)}",
            "side_of_anchor", (f"anchor-right:{anchor_index}:{anchor_category}",),
        ))
    return out


AREA_RATIO_MIN = 1.5


def _realize_attribute(f: ObjectFacts) -> list[Realization]:
    out: list[Realization] = []
    head = f.head
    color_ok = bool(f.color) and head not in COLOR_SUPPRESSED_HEADS
    if color_ok and f.features and " " not in f.features and not ACTION_FEATURE_RE.match(f.features):
        out.append(_variant(f"The {f.color} {f.features} {head}", "plain_attribute", ("color", "feature")))
    if color_ok and f.features and ACTION_FEATURE_RE.match(f.features) is None:
        out.append(_variant(f"The {color_ok and f.color or ''} {head} with {f.features}".replace("  ", " "),
                            "plain_attribute", ("color", "feature")))
    if color_ok:
        out.append(_variant(f"The {f.color} {head}", "plain_attribute", ("color",)))
        out.append(_variant(f"{article_for(head).capitalize()} {f.color} {head}", "plain_attribute", ("color",)))
    # Size comparative (admin rule): the object leads its same-head group in
    # bbox area by a strict ratio. "larger" needs a rival; "largest" needs 3+.
    if f.area_ratio_lead is not None and f.area_ratio_lead >= AREA_RATIO_MIN:
        if f.count_in_head >= 2:
            out.append(_variant(
                f"The larger {f'{f.color} ' if color_ok else ''}{head}".replace("  ", " "),
                "plain_attribute", ("area-comparative",),
            ))
        if f.count_in_head >= 3:
            out.append(_variant(
                f"The largest {f'{f.color} ' if color_ok else ''}{head}".replace("  ", " "),
                "plain_attribute", ("area-superlative",),
            ))
    if f.features and ACTION_FEATURE_RE.match(f.features):
        out.append(_variant(f"The {head} {f.features}", "action_feature", ("feature",)))
    elif f.features:
        out.append(_variant(f"The {head} with {f.features}", "plain_attribute", ("feature",)))
    elif f.count_in_head == 1:
        out.append(_variant(f"The {head}", "plain_attribute", ("unique-category",)))
    return out


def realizations_for(facts: ObjectFacts) -> list[Realization]:
    """All fact-supported variants for one object, deterministic order."""
    return (
        _realize_ordinal(facts)
        + _realize_superlative(facts)
        + _realize_anchor(facts)
        + _realize_attribute(facts)
    )


def realization_is_unique(
    realization: Realization, frame: list[ObjectFacts], target: ObjectFacts
) -> bool:
    """Code-verifiable uniqueness: the description resolves to one object."""
    family = realization.family
    if family == "ordinal_direction":
        return target.rank_left is not None or target.rank_right is not None
    if family == "plain_attribute" and "area-comparative" in realization.facts or "area-superlative" in realization.facts:
        return True  # strict area ratio holds for at most one object per head group
    if family == "superlative_camera":
        if realization.facts[0] in ("foreground", "background"):
            # Band membership is not exclusive: disambiguation requires the
            # target to be the only same-head object in that band.
            if realization.facts[0] == "foreground":
                return sum(1 for o in frame if o.head == target.head and o.is_in_foreground) == 1
            return sum(1 for o in frame if o.head == target.head and o.is_in_background) == 1
        return True  # each exclusive flag holds for at most one object per frame
    if family == "side_of_anchor":
        text = realization.text.lower()
        if "side of the image" in text:
            side = "left" if "on the left side" in text else "right"
            return sum(1 for o in frame if o.head == target.head and o.side_of_image == side) == 1
        anchor_parts = realization.facts[0].split(":")
        anchor_index = int(anchor_parts[1])
        direction = "left" if anchor_parts[0] == "anchor-left" else "right"
        anchor = next((o for o in frame if o.index == anchor_index), None)
        if anchor is None:
            return False
        anchor_head = category_head(anchor.category)
        if sum(1 for o in frame if o.head == anchor_head) != 1:
            return False

        def related(o: ObjectFacts) -> bool:
            if direction == "left":
                return anchor.bbox[0] - o.bbox[2] >= ANCHOR_GAP
            return o.bbox[0] - anchor.bbox[2] >= ANCHOR_GAP

        return sum(1 for o in frame if o.head == target.head and related(o)) == 1
    if family in ("plain_attribute", "action_feature"):
        # Objects with unknown attributes count as potential matches: a missing
        # attr entry never licenses a uniqueness claim.
        def matches(o: ObjectFacts) -> bool:
            if o.head != target.head:
                return False
            if "color" in realization.facts and o.color is not None and o.color != target.color:
                return False
            if "feature" in realization.facts and o.features is not None and o.features != target.features:
                return False
            return True

        return sum(1 for o in frame if matches(o)) == 1
    return False


@dataclass
class AssemblyRecord:
    sample_id: str
    sequence_id: str
    source: str  # "real" | "teacher"
    category: str
    bbox: list[float]
    object_index: int
    query: str
    family: str
    bucket: str
    quota_state: str  # "quota" | "overshoot"
    facts: list[str]
    words: int


@dataclass
class AssemblyResult:
    records: list[AssemblyRecord] = field(default_factory=list)
    shortfall: list[dict] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)


def _with_color(f: ObjectFacts, color: str | None) -> ObjectFacts:
    return replace(f, color=color)


def select_targets(
    frame_facts: list[ObjectFacts], *, max_teacher: int = 2
) -> list[tuple[str, ObjectFacts]]:
    """Real target (canary) first, then best teacher objects by attr quality."""
    targets: list[tuple[str, ObjectFacts]] = []
    canary = next((f for f in frame_facts if f.is_canary), None)
    if canary is not None:
        targets.append(("real", canary))
    teachers = [
        f for f in frame_facts
        if not f.is_canary and MIN_TEACHER_AREA <= f.area <= MAX_TEACHER_AREA
    ]
    teachers.sort(
        key=lambda f: (
            2 if f.color and f.features else 1 if f.color or f.features else 0,
            f.count_in_head >= 2,
            -f.area,
        ),
        reverse=True,
    )
    if max_teacher < 0:
        targets.extend(("teacher", f) for f in teachers)  # -1 = uncapped
    else:
        targets.extend(("teacher", f) for f in teachers[:max_teacher])
    return targets


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


class _SupplyItem:
    """One allocatable target with its pre-verified realization variants."""

    __slots__ = ("sample_id", "sequence_id", "source", "facts", "gt_bbox",
                 "variants", "depth_available", "ordinal_allowed")

    def __init__(self, sample_id, sequence_id, source, facts, gt_bbox, variants,
                 depth_available, ordinal_allowed=True):
        self.sample_id = sample_id
        self.sequence_id = sequence_id
        self.source = source
        self.facts = facts
        self.gt_bbox = gt_bbox
        self.variants = variants
        self.depth_available = depth_available
        self.ordinal_allowed = ordinal_allowed


def assemble_run(
    merged: dict,
    index: dict,
    spec: dict | None = None,
    *,
    enumeration: dict | None = None,
    max_teacher_per_frame: int = 2,
    min_words: int = 3,
    max_words: int = 18,
) -> AssemblyResult:
    """Assemble query records from a census merged.json + dataset index.

    Deterministic: same inputs -> same records. Supply (targets + their
    fact-supported, uniqueness-verified realizations) is built first, then
    the Phase 3 planner allocates under the frozen spec quotas. A target
    that cannot produce a unique realization is reported as shortfall —
    never fabricated.
    """
    result = AssemblyResult()
    sequences = merged.get("results", {})
    supply: list[_SupplyItem] = []

    for sequence_id in sorted(sequences):
        seq = sequences[sequence_id]
        if seq.get("status") != "completed":
            for sample_id in sorted(seq.get("selected") or []):
                result.shortfall.append(
                    {"sample_id": sample_id, "reason": "sequence-not-completed"}
                )
            continue
        for sample_id in sorted(seq.get("selected") or []):
            frame = seq["frames"].get(sample_id)
            if not frame or frame.get("status") != "completed":
                result.shortfall.append({"sample_id": sample_id, "reason": "frame-not-completed"})
                continue
            entry = index.get(sample_id)
            if entry is None:
                result.shortfall.append({"sample_id": sample_id, "reason": "sample-missing-from-index"})
                continue
            if frame.get("single_pass"):
                good = (
                    frame["findall_1"]
                    if frame["findall_1"]["status"] == "completed"
                    else frame["findall_2"]
                )
                objects = good["objects"]
            else:
                objects = pass_agreement(
                    frame["findall_1"]["objects"], frame["findall_2"]["objects"]
                )["agreed_objects"]
            if not objects:
                result.shortfall.append({"sample_id": sample_id, "reason": "no-agreed-objects"})
                continue
            # Enumeration-pass re-ranking: capped frames get scene-true
            # object sets (up) or phantom-pruned sets (down); inconsistent
            # frames keep the original set but lose ordinal privileges.
            frame_verdict = None
            if enumeration:
                frame_verdict = verdict_for(sample_id, frame, enumeration)
                if frame_verdict.verdict == "up" and frame_verdict.real_objects:
                    objects = frame_verdict.real_objects
                elif frame_verdict.verdict == "down" and frame_verdict.real_objects:
                    # Phantom-pruned: the agreed subset surviving in the full
                    # re-enumeration. Match originals against it by IoU.
                    from foundry.bbox import compute_iou as _ci

                    kept = [
                        o for o in objects
                        if any(_ci(o["bbox"], ro["bbox"]) >= 0.5 for ro in frame_verdict.real_objects)
                    ]
                    if kept:
                        objects = kept
            facts = extract_frame_facts(objects, entry["bbox"], frame.get("attr"), frame.get("depth"))

            # Color arbitration (admin 2026-09-06): a color claimed by 2+
            # same-head objects has no referential power in this frame —
            # strip it from realizations for everyone in the head group.
            head_colors: dict[str, Counter] = {}
            for f in facts:
                if f.color:
                    head_colors.setdefault(f.head, Counter())[f.color] += 1
            facts = [
                _with_color(f, None)
                if f.color and head_colors.get(f.head, {}).get(f.color, 0) >= 2
                else f
                for f in facts
            ]
            if not any(f.is_canary for f in facts):
                result.shortfall.append(
                    {"sample_id": sample_id, "reason": "no-canary-in-agreed-objects"}
                )
            result.sequences.append(sequence_id)
            depth_available = (frame.get("depth") or {}).get("source") == "raw-uint16-mm"
            # Cap-truncation gate (admin 2026-09-06): a frame at the census
            # cap likely truncated its enumeration, so bare ordinals stop
            # being scene-consistent. Edge ranks (1st/last) survive; the
            # red-boxed category is filled first per the prompt and keeps
            # its ordinals too.
            frame_capped = len(objects) >= 6 or (
                frame_verdict is not None and frame_verdict.verdict == "inconsistent"
            )
            canary_head = next(
                (f.head for f in facts if f.is_canary), None
            )
            for source, target_facts in select_targets(facts, max_teacher=max_teacher_per_frame):
                edge_rank = (
                    target_facts.rank_left in (1, target_facts.count_in_head)
                    or target_facts.rank_right == 1
                    or target_facts.rank_right == target_facts.count_in_head
                )
                priority_cat = canary_head is not None and target_facts.head == canary_head
                ordinal_allowed = (
                    not frame_capped or edge_rank or priority_cat or target_facts.is_canary
                )
                variants = [
                    r for r in realizations_for(target_facts)
                    if min_words <= r.words <= max_words
                    and realization_is_unique(r, facts, target_facts)
                ]
                if not ordinal_allowed:
                    variants = [v for v in variants if v.family != "ordinal_direction"]
                supply.append(_SupplyItem(
                    sample_id=sample_id,
                    sequence_id=sequence_id,
                    source=source,
                    facts=target_facts,
                    gt_bbox=list(entry["bbox"]),
                    variants=variants,
                    depth_available=depth_available,
                    ordinal_allowed=ordinal_allowed,
                ))

    target_supplies = [
        TargetSupply(
            sample_id=item.sample_id,
            sequence_id=item.sequence_id,
            source=item.source,
            facts=item.facts,
            gt_bbox=item.gt_bbox,
            realizations=item.variants,
            depth_available=item.depth_available,
        )
        for item in supply
    ]
    plan_result = planner_plan(target_supplies, spec)
    for allocation in plan_result.allocations:
        chosen = allocation.realization
        item = allocation.supply
        result.records.append(
            AssemblyRecord(
                sample_id=item.sample_id,
                sequence_id=item.sequence_id,
                source=item.source,
                category=item.facts.category,
                # The real target trains on the organizer GT box itself; the
                # enumerated canary box only supplies its facts.
                bbox=list(item.gt_bbox) if item.source == "real" else list(item.facts.bbox),
                object_index=item.facts.index,
                query=chosen.text,
                family=chosen.family,
                bucket=allocation.bucket,
                quota_state=allocation.quota_state,
                facts=list(chosen.facts),
                words=chosen.words,
            )
        )
    result.shortfall.extend(plan_result.unallocated)
    result.sequences = sorted(set(result.sequences))
    return result


def audit_assembly(records: list[AssemblyRecord]) -> dict:
    """Acceptance metrics: repeat rate, bucket shares, word counts, sources."""
    total = len(records)
    if not total:
        raise ValueError("No assembled records to audit")
    texts = [r.query for r in records]
    verbatim = len(set(texts))
    lower = len(set(t.lower() for t in texts))
    buckets = {bucket: 0 for bucket in FROZEN_BUCKETS}
    for r in records:
        buckets[r.bucket] += 1
    words = [r.words for r in records]
    sources = {"real": 0, "teacher": 0}
    for r in records:
        sources[r.source] += 1
    repeat_rate = 1 - verbatim / total
    return {
        "count": total,
        "verbatim_unique": verbatim,
        "verbatim_repeat_rate": round(repeat_rate, 4),
        "lower_repeat_rate": round(1 - lower / total, 4),
        "bucket_counts": buckets,
        "bucket_per_mille": {b: round(c / total * 1000) for b, c in buckets.items()},
        "mean_words": round(sum(words) / total, 2),
        "median_words": sorted(words)[total // 2],
        "sources": sources,
        "overshoot_records": sum(1 for r in records if r.quota_state == "overshoot"),
        "test_reference": {
            "verbatim_repeat_rate": 0.076,
            "mean_words": 10.33,
            "bucket_per_mille": {
                "ordinal": 335, "spatial": 258, "attribute_action": 256, "distance": 151,
            },
        },
        "acceptance": {"verbatim_repeat_le_0_10": repeat_rate <= 0.10},
    }
