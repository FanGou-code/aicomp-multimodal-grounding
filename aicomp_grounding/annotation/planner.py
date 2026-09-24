"""Phase 3 planner: turn per-target candidates into the corpus.

Each target carries the one sentence the teacher wrote from the facts code
listed. The planner keeps one sentence per target, refuses to emit the same
sentence twice in a run, and reports what it could not use. What the corpus
ends up looking like is whatever the frames and the teacher produce: nothing
here steers composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aicomp_grounding.annotation.facts import ObjectFacts, Realization


@dataclass
class TargetSupply:
    sample_id: str
    sequence_id: str
    source: str  # "real" | "teacher"
    facts: ObjectFacts
    gt_bbox: list[float]
    realizations: list[Realization]


@dataclass
class Allocation:
    supply: TargetSupply
    realization: Realization


@dataclass
class PlanResult:
    allocations: list[Allocation] = field(default_factory=list)
    unallocated: list[dict] = field(default_factory=list)


def plan(supply: list[TargetSupply]) -> PlanResult:
    """One sentence per target, deduplicated across the run."""
    result = PlanResult()
    used_texts: set[str] = set()

    for target in supply:
        variants = [r for r in target.realizations if r.text.lower() not in used_texts]
        if not variants:
            result.unallocated.append(
                {"sample_id": target.sample_id, "source": target.source,
                 "reason": "sentence-already-used"}
            )
            continue
        chosen = variants[0]
        used_texts.add(chosen.text.lower())
        result.allocations.append(Allocation(supply=target, realization=chosen))
    return result
