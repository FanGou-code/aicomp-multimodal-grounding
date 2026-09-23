"""Local query assembly from census facts.

For each selected target, code picks the dimension that disambiguates it inside
its own frame (``foundry.pipeline.realize``), the teacher turns that dimension
into one sentence, and code verifies the sentence names the facts it was handed.
The planner then allocates under the frozen spec quotas.

Head groups are keyed on the category HEAD noun (last word): "swan" and
"black swan" share the head "swan" and are one group for ranks and uniqueness.
Geometric facts are computed in ``foundry.pipeline.facts``.

The bucket classifier is imported from the frozen Phase 0 mining script so
acceptance shares stay byte-identical with the published test-side counts
(ordinal > distance > spatial > attribute_action).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from foundry.pipeline.census import trusted_objects
from foundry.pipeline.facts import (
    ObjectFacts,
    Realization,
    extract_frame_facts,
)
from foundry.pipeline.planner import TargetSupply, plan as planner_plan
from foundry.pipeline.depth import DEPTH_SOURCE
from foundry.pipeline.buckets import FROZEN_BUCKETS
from foundry.pipeline.realize import choose_dimension

MIN_TEACHER_AREA = 0.002
MAX_TEACHER_AREA = 0.6


def _variant(text: str, family: str, facts: tuple[str, ...]) -> Realization:
    return Realization(" ".join(text.split()), family, facts, len(text.split()))


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
    edited: bool = False  # text QC modified the query (foundry.text_qc)


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
    """The reference object first when the enumeration re-found it, then the
    best teacher objects by attribute quality.
    """
    targets: list[tuple[str, ObjectFacts]] = []
    reference = next((f for f in frame_facts if f.is_canary), None)
    if reference is not None:
        targets.append(("real", reference))
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


class _SupplyItem:
    """One allocatable target with its pre-verified realization variants."""

    __slots__ = ("sample_id", "sequence_id", "source", "facts", "gt_bbox",
                 "variants", "depth_available")

    def __init__(self, sample_id, sequence_id, source, facts, gt_bbox, variants,
                 depth_available):
        self.sample_id = sample_id
        self.sequence_id = sequence_id
        self.source = source
        self.facts = facts
        self.gt_bbox = gt_bbox
        self.variants = variants
        self.depth_available = depth_available



def assemble_run(
    merged: dict,
    index: dict,
    spec: dict | None = None,
    *,
    max_teacher_per_frame: int = 2,
    realize=None,
) -> AssemblyResult:
    """Assemble query records from a census merged.json + dataset index.

    ``realize`` is a callable ``(dimension, bbox, sample_id) -> str | None``.
    Whatever it returns goes to human review unjudged; a target with no usable
    dimension, or whose reply carried no sentence, is reported as shortfall.
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
            objects = trusted_objects(frame)
            if not objects:
                result.shortfall.append({"sample_id": sample_id, "reason": "no-enumerated-objects"})
                continue
            facts = extract_frame_facts(objects, entry["bbox"], frame.get("attr"), frame.get("depth"))

            # Color arbitration: a color claimed by 2+ same-head objects has no
            # referential power in this frame — strip it from everyone in the
            # head group so no dimension is built on it.
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
            result.sequences.append(sequence_id)
            depth_available = (frame.get("depth") or {}).get("source") == DEPTH_SOURCE
            for source, target_facts in select_targets(facts, max_teacher=max_teacher_per_frame):
                dimension = choose_dimension(target_facts, facts)
                if dimension is None:
                    result.shortfall.append(
                        {"sample_id": sample_id, "reason": "no-dimension"}
                    )
                    continue
                sentence = (
                    realize(dimension, list(target_facts.bbox), sample_id)
                    if realize is not None
                    else None
                )
                if sentence is None:
                    result.shortfall.append(
                        {"sample_id": sample_id, "reason": "no-sentence"}
                    )
                    continue
                supply.append(_SupplyItem(
                    sample_id=sample_id,
                    sequence_id=sequence_id,
                    source=source,
                    facts=target_facts,
                    gt_bbox=list(entry["bbox"]),
                    variants=[_variant(
                        sentence, dimension["family"], tuple(dimension["facts"])
                    )],
                    depth_available=depth_available,
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
        "acceptance": {"verbatim_repeat_le_0_10": repeat_rate <= 0.10},
    }
