"""Local query generation from census facts.

For each selected target, code writes out every fact that is true of it inside
its own frame (``foundry.pipeline.realize``), the teacher turns those facts
into one sentence, and code verifies the sentence names the facts it was handed.
Every sentence the teacher returns is emitted as-is; the planner only
deduplicates.

Head groups are keyed on the category HEAD noun (last word): "swan" and
"black swan" share the head "swan" and are one group for ranks and uniqueness.
Geometric facts are computed in ``foundry.pipeline.facts``.

How the corpus is composed is whatever the frames and the teacher produce;
nothing here steers it, and no share of any kind is computed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from aicomp_grounding.annotation.census import trusted_objects
from aicomp_grounding.annotation.facts import (
    ObjectFacts,
    Realization,
    extract_frame_facts,
)
from aicomp_grounding.annotation.selection import TargetSupply, plan as planner_plan

MIN_TEACHER_AREA = 0.002
MAX_TEACHER_AREA = 0.6


def _variant(text: str) -> Realization:
    return Realization(" ".join(text.split()), len(text.split()))


@dataclass
class GenerationRecord:
    sample_id: str
    sequence_id: str
    source: str  # "real" | "teacher"
    category: str
    bbox: list[float]
    object_index: int
    query: str
    words: int


@dataclass
class GenerationResult:
    records: list[GenerationRecord] = field(default_factory=list)
    shortfall: list[dict] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)



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
                 "variants")

    def __init__(self, sample_id, sequence_id, source, facts, gt_bbox, variants):
        self.sample_id = sample_id
        self.sequence_id = sequence_id
        self.source = source
        self.facts = facts
        self.gt_bbox = gt_bbox
        self.variants = variants



@dataclass(frozen=True)
class _PendingTarget:
    """One selected target, before its wording call."""

    sample_id: str
    sequence_id: str
    source: str
    facts: ObjectFacts
    gt_bbox: list[float]

    @property
    def item_id(self) -> str:
        """Stable id of this target across runs of the same census."""
        return f"{self.sample_id}#{self.facts.index:02d}"


def generate_run(
    merged: dict,
    index: dict,
    *,
    max_teacher_per_frame: int = 2,
    max_workers: int = 8,
    realize=None,
    sentences: dict[str, str | None] | None = None,
    on_sentence=None,
) -> GenerationResult:
    """Generate query records from a census merged.json + dataset index.

    ``realize`` is a callable ``(facts, sample_id) -> str | None``, called once
    per selected target that has no wording yet. Everything that can be computed
    locally happens first, then the calls run on ``max_workers`` threads.
    Whatever they return goes to human review unjudged; a target whose reply
    carried no sentence is reported as shortfall.

    ``sentences`` carries wording obtained earlier, keyed by ``item_id``; those
    targets are not called again, and it is updated in place with the fresh
    results. ``on_sentence(item_id, sentence)`` is invoked on the calling thread
    as each fresh result arrives, which is where a caller persists it if it
    wants a resumable run.
    """
    result = GenerationResult()
    sequences = merged.get("results", {})
    supply: list[_SupplyItem] = []
    pending: list[_PendingTarget] = []

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

            result.sequences.append(sequence_id)
            for source, target_facts in select_targets(facts, max_teacher=max_teacher_per_frame):
                pending.append(_PendingTarget(
                    sample_id=sample_id,
                    sequence_id=sequence_id,
                    source=source,
                    facts=target_facts,
                    gt_bbox=list(entry["bbox"]),
                ))

    done = sentences if sentences is not None else {}
    todo = [item for item in pending if item.item_id not in done]

    def record(item: _PendingTarget, sentence: str | None) -> None:
        done[item.item_id] = sentence
        if on_sentence is not None:
            on_sentence(item.item_id, sentence)

    if realize is not None and todo:
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(realize, item.facts, item.sample_id): item
                    for item in todo
                }
                for future in as_completed(futures):
                    record(futures[future], future.result())
        else:
            for item in todo:
                record(item, realize(item.facts, item.sample_id))

    for item in pending:
        sentence = done.get(item.item_id)
        if sentence is None:
            result.shortfall.append({"sample_id": item.sample_id, "reason": "no-sentence"})
            continue
        supply.append(_SupplyItem(
            sample_id=item.sample_id,
            sequence_id=item.sequence_id,
            source=item.source,
            facts=item.facts,
            gt_bbox=item.gt_bbox,
            variants=[_variant(sentence)],
        ))

    target_supplies = [
        TargetSupply(
            sample_id=item.sample_id,
            sequence_id=item.sequence_id,
            source=item.source,
            facts=item.facts,
            gt_bbox=item.gt_bbox,
            realizations=item.variants,
        )
        for item in supply
    ]
    plan_result = planner_plan(target_supplies)
    for allocation in plan_result.allocations:
        chosen = allocation.realization
        item = allocation.supply
        result.records.append(
            GenerationRecord(
                sample_id=item.sample_id,
                sequence_id=item.sequence_id,
                source=item.source,
                category=item.facts.category,
                # The real target trains on the organizer GT box itself; the
                # enumerated canary box only supplies its facts.
                bbox=list(item.gt_bbox) if item.source == "real" else list(item.facts.bbox),
                object_index=item.facts.index,
                query=chosen.text,
                words=chosen.words,
            )
        )
    result.shortfall.extend(plan_result.unallocated)
    result.sequences = sorted(set(result.sequences))
    return result


def audit_generation(records: list[GenerationRecord]) -> dict:
    """Acceptance metrics: repeat rate, word counts, sources."""
    total = len(records)
    if not total:
        raise ValueError("No generated records to audit")
    texts = [r.query for r in records]
    verbatim = len(set(texts))
    lower = len(set(t.lower() for t in texts))
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
        "mean_words": round(sum(words) / total, 2),
        "median_words": sorted(words)[total // 2],
        "sources": sources,
        "acceptance": {"verbatim_repeat_le_0_10": repeat_rate <= 0.10},
    }
