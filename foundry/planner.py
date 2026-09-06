"""Phase 3 planner: quota-guided sentence-pattern allocation (规划器只分).

Consumes the assembler's supply (per-target fact-supported, uniqueness-verified
realizations) and decides which realization each target emits. Deterministic:
same inputs -> same allocations. Policy, all admin-approved:

- Bucket quotas follow the frozen style-spec shares (ordinal 335 / spatial 258
  / attribute_action 256 / distance 151 per-mille); targets whose facts cannot
  fill their assigned bucket fall through the others, and impossible targets
  are reported as unallocated — never fabricated. Verbatim texts are unique
  across the whole run.
- Crowd disambiguation: when a frame carries >= 3 same-head objects AND its
  depth facts are available, band realizations (in the foreground / in the
  background) get first pick of the distance bucket — scene-crowd ambiguity
  is what depth words resolve.
- Richness: within a bucket the longest realization wins, with a diversity
  penalty on families already used in the same sequence so surface forms
  spread instead of stacking.
- Overshoot is allowed (and flagged) rather than dropping a usable target.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from foundry.facts import ObjectFacts, Realization
from foundry.buckets import FROZEN_BUCKETS, classify_frozen, parse_spec_shares

CROWD_SAME_HEAD = 3


@dataclass
class TargetSupply:
    sample_id: str
    sequence_id: str
    source: str  # "real" | "teacher"
    facts: ObjectFacts
    gt_bbox: list[float]
    realizations: list[Realization]
    depth_available: bool = False
    ordinal_allowed: bool = True  # capped frames suppress mid-rank ordinals


@dataclass
class Allocation:
    supply: TargetSupply
    realization: Realization
    bucket: str
    quota_state: str  # "quota" | "overshoot"


@dataclass
class PlanResult:
    allocations: list[Allocation] = field(default_factory=list)
    unallocated: list[dict] = field(default_factory=list)


def _pick(variants: list[Realization], sequence_id: str, family_use: dict) -> Realization:
    """Longest first, then least-reused family in the sequence, then stable text."""
    return max(
        variants,
        key=lambda r: (r.words, -family_use[(sequence_id, r.family)], r.text),
    )


def plan(supply: list[TargetSupply], spec: dict | None) -> PlanResult:
    """Allocate one realization per supplied target under frozen quotas."""
    result = PlanResult()
    shares = parse_spec_shares(spec)
    per_mille_total = sum(shares.values())
    remaining = {
        bucket: round(len(supply) * shares[bucket] / per_mille_total)
        for bucket in FROZEN_BUCKETS
    }
    priority = {bucket: rank for rank, bucket in enumerate(FROZEN_BUCKETS)}
    family_use: dict[tuple[str, str], int] = defaultdict(int)
    used_texts: set[str] = set()

    for target in supply:
        variants = [
            r for r in target.realizations
            if r.text.lower() not in used_texts
            and (target.ordinal_allowed or r.family != "ordinal_direction")
        ]
        if not variants:
            result.unallocated.append(
                {"sample_id": target.sample_id, "source": target.source,
                 "reason": "no-unique-realization"}
            )
            continue

        bucket_order = sorted(FROZEN_BUCKETS, key=lambda b: (-remaining[b], priority[b]))
        # Crowd rule: same-head crowding is what depth bands disambiguate, so
        # the buckets holding band realizations get first pick for these
        # targets. (Under the frozen classifier "in the foreground/background"
        # lands in attribute_action, not distance - so resolve the actual
        # buckets dynamically instead of assuming.)
        band_buckets = {
            classify_frozen(r.text)
            for r in variants
            if r.facts and r.facts[0] in ("foreground", "background")
        }
        crowd = (
            target.depth_available
            and target.facts.count_in_head >= CROWD_SAME_HEAD
            and band_buckets
        )
        if crowd:
            preferred = [b for b in bucket_order if b in band_buckets]
            rest = [b for b in bucket_order if b not in band_buckets]
            bucket_order = preferred + rest

        chosen: Realization | None = None
        chosen_bucket: str | None = None
        state = "quota"
        for bucket in bucket_order:
            if remaining[bucket] <= 0:
                continue
            in_bucket = [r for r in variants if classify_frozen(r.text) == bucket]
            if not in_bucket:
                continue
            chosen = _pick(in_bucket, target.sequence_id, family_use)
            chosen_bucket = bucket
            break
        if chosen is None:
            # Quotas exhausted for every supportable bucket: keep the target,
            # flag the overshoot instead of wasting supply.
            chosen = _pick(variants, target.sequence_id, family_use)
            chosen_bucket = classify_frozen(chosen.text)
            state = "overshoot"

        remaining[chosen_bucket] -= 1
        family_use[(target.sequence_id, chosen.family)] += 1
        used_texts.add(chosen.text.lower())
        result.allocations.append(
            Allocation(
                supply=target,
                realization=chosen,
                bucket=chosen_bucket,
                quota_state=state,
            )
        )
    return result
