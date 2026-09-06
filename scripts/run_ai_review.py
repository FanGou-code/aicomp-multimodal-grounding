#!/usr/bin/env python3
"""AI pre-review pass: audit assembled queries against the scene before human review.

For every assembled record (one query + its target), render the full scene with
the target marked by a thick red rectangle -- in memory only, zero images on
disk -- and ask the teacher to audit it: does the query point at the target, is
the ordinal exact, do color/feature words match, is the description unique? The
teacher may correct ONLY the ordinal word, the direction phrase, or a visibly
wrong color/feature word; anything else goes to the human queue.

    {item_id: {"verdict": "pass"|"fixed"|"human", "query": str, "observed": str,
               "reason": str, "attempts": int, "status": "completed"|"failed"}}

The merged ``ai_review.json`` is written next to the shards. Apply-side gates
(one per fixed query: uniqueness matcher, bucket classifier, text QC) live in
the apply step, not here. Resumable: rerunning skips finished items; items with
a human verdict in the review store are skipped entirely (human work is never
touched). Every real API call requires the administrator's explicit go.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.api_client import (  # noqa: E402
    APIError,
    OpenAIProtocolClient,
    SlidingWindowRateLimiter,
)
from foundry.annotation_views import build_marked_annotation_view, jpeg_data_url  # noqa: E402
from foundry.census import _parse_json_object  # noqa: E402
from foundry.config import (  # noqa: E402
    ANNOTATION_API_BASE_URL,
    ANNOTATION_MODEL_NAME,
    ANNOTATION_REQUESTS_PER_MINUTE,
    ANNOTATION_TEMPERATURE,
    ANNOTATION_TOKENS_PER_MINUTE,
)
from foundry.io import atomic_write_json, load_json  # noqa: E402
from foundry.keys import (  # noqa: E402
    APIKeyPool,
    DEFAULT_KEY_FILE,
    comment_out_key,
    load_api_keys,
)
from foundry.review.census_session import TEACHER_ANNOTATOR  # noqa: E402
from scripts.run_census import GENERATION_CONFIG  # noqa: E402

AI_REVIEW_PROTOCOL = "ai-pre-review-v1"
AI_REVIEW_PROMPT = """You are auditing one assembled referring query for a visual-grounding dataset.

The image is the full scene. The TARGET is the object marked with a thick red rectangle. The proposed query is:
"<query>"

Audit it in this order:
1. Look at the red-boxed target: what is it, and what are its visible attributes (color, notable features)?
2. TARGET: does the query point at the red-boxed object (never another one)?
3. ORDINAL: if the query contains an ordinal with a direction (from left to right / from right to left), count the same-category objects as a careful human would — clearly visible, confidently nameable instances only; ignore tiny, blurry or ambiguous clutter. The ordinal word must be exact. If counting along the stated direction is unreliable (crowded or overlapping on that side), flip the direction (left to right <-> right to left) and recompute the ordinal word instead.
4. COLOR / FEATURE: every color or visible-feature word must match the target.
5. UNIQUENESS: among confidently identifiable same-category objects, no other object may match the query as well as the target does. Two equally good matches = ambiguous.

You may correct the query by changing ONLY:
- the ordinal word (first/second/third/...),
- the direction phrase (from left to right <-> right to left),
- a visibly wrong color or feature word.
A correction is an edit, never a rewrite: every other word stays exactly as-is. Never restructure the sentence, never add or drop information, never change the target.

6. RE-CHECK YOUR EDIT: the corrected query must still resolve to exactly one object — the target. If your edited version matches more than one object, do not ship it: output verdict "human" instead.

Output JSON only:
{"observed": "<what you counted / saw, <=20 words>",
 "verdict": "pass" | "fixed" | "human",
 "query": "<final query; identical to the input when verdict=pass>",
 "reason": "<<=15 words: what you fixed, or why a human is needed>"}

Use "human" when: you are not confident in the count, two objects match equally, the target itself looks wrong, or the fix needs anything beyond the three allowed edits."""

AUDIT_MAX_TOKENS = 512
ATTEMPTS_PER_ITEM = 2
VALID_VERDICTS = ("pass", "fixed", "human")


def prompt_hash() -> str:
    import hashlib

    return hashlib.sha256(AI_REVIEW_PROMPT.encode("utf-8")).hexdigest()


def item_id_of(record: dict) -> str:
    return f"{record['sample_id']}#{record['object_index']:02d}"


def normalize_query(query: str) -> str:
    return " ".join(query.split()).rstrip(".?!;")


def parse_audit_response(text: object, original_query: str) -> dict:
    """Validate one audit response; coerce pass/fixed inconsistencies."""
    payload = _parse_json_object(text, label="AI review")
    if set(payload) != {"observed", "verdict", "query", "reason"}:
        raise ValueError("AI review schema must be exactly {observed, verdict, query, reason}")
    verdict = payload["verdict"]
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {verdict!r}")
    for field in ("observed", "reason"):
        if not isinstance(payload[field], str) or len(payload[field]) > 200:
            raise ValueError(f"{field} must be a short string")
    query = normalize_query(str(payload["query"]))
    if not query or len(query) > 200:
        raise ValueError("query must be a non-empty string (max 200 chars)")
    coerced = False
    if verdict == "pass" and query != original_query:
        verdict, coerced = "fixed", True
    elif verdict == "fixed" and query == original_query:
        verdict, coerced = "pass", True
    return {
        "observed": payload["observed"].strip(),
        "verdict": verdict,
        "query": query,
        "reason": payload["reason"].strip(),
        "coerced": coerced,
    }


def audit_messages(marked_jpeg_url: str, query: str, *, previous_error: str = "") -> list[dict]:
    repair = ""
    if previous_error:
        repair = (
            "\nThe previous response failed deterministic validation for this reason: "
            f"{previous_error}. Correct that failure and regenerate the complete JSON object."
        )
    return [
        {"role": "system", "content": "You are a meticulous visual-grounding auditor. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "One complete RGB scene; the thick red rectangle marks the TARGET object to audit."},
                {"type": "image_url", "image_url": {"url": marked_jpeg_url, "detail": "high"}},
                {"type": "text", "text": AI_REVIEW_PROMPT.replace("<query>", query) + repair},
            ],
        },
    ]


def human_reviewed_ids(review_root: Path, run_tag: str) -> set[str]:
    """Item ids carrying a human verdict in the review store (never touched)."""
    meta = {}
    store_dir = Path(review_root) / run_tag
    journal = store_dir / "annotations.jsonl"
    if not journal.is_file():
        return set()
    with journal.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            annotator = record.get("annotator")
            if annotator and annotator != TEACHER_ANNOTATOR:
                meta[record.get("id")] = True
    return {item_id for item_id in meta if item_id}


def build_plan(assembly_path: Path, data_root: Path, review_root: Path) -> dict:
    """Todo items for the pass: assembly records minus human-reviewed ones."""
    manifest = load_json(Path(assembly_path))
    metadata = manifest.get("metadata", {})
    run_tag = str(metadata.get("run_tag") or Path(assembly_path).parent.name)
    split = metadata.get("split", "train")
    index = load_json(Path(data_root) / "indexes" / f"{split}.json")
    reviewed = human_reviewed_ids(Path(review_root), run_tag)

    items: list[dict] = []
    skipped_human = 0
    for record in manifest.get("records", []):
        item_id = item_id_of(record)
        if item_id in reviewed:
            skipped_human += 1
            continue
        entry = index.get(record["sample_id"])
        if entry is None:
            continue
        items.append({
            "item_id": item_id,
            "query": record["query"],
            "bbox": record["bbox"],
            "visible": entry["visible"],
            "source": record["source"],
        })
    return {
        "run_tag": run_tag,
        "split": split,
        "items": sorted(items, key=lambda it: it["item_id"]),
        "skipped_human": skipped_human,
        "records_total": len(manifest.get("records", [])),
    }


def _load_plain(data_root: Path, visible: str):
    from PIL import Image

    with Image.open(Path(data_root) / visible) as img:
        return img.convert("RGB")


def audit_item(client, item: dict, data_root: Path) -> dict:
    """One audit call with repair attempts; deterministic parse validation."""
    last_error = ""
    calls = 0
    for attempt in range(1, ATTEMPTS_PER_ITEM + 1):
        try:
            plain = _load_plain(data_root, item["visible"])
            marked = jpeg_data_url(build_marked_annotation_view(plain, item["bbox"]))
            response = client.complete(
                messages=audit_messages(marked, item["query"],
                                        previous_error=last_error if attempt > 1 else ""),
                max_tokens=AUDIT_MAX_TOKENS,
                temperature=ANNOTATION_TEMPERATURE,
            )
            calls += 1
            parsed = parse_audit_response(response.content, item["query"])
            return {
                "status": "completed",
                "attempts": attempt,
                "calls": calls,
                **parsed,
            }
        except (APIError, ValueError) as exc:
            last_error = str(exc)[:200]
            calls += 1
    return {"status": "failed", "attempts": ATTEMPTS_PER_ITEM, "calls": calls,
            "error": last_error}


def review_shard(
    shard_id: int,
    items: list[dict],
    data_root: Path,
    output_root: Path,
    run_tag: str,
    key_pool,
    rate_limiter,
    *,
    timeout_seconds: float,
) -> dict:
    results: dict = {}
    out_dir = output_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / "shards" / f"shard_{shard_id:02d}.json"
    if shard_path.exists():
        results = load_json(shard_path).get("results", {})

    client = OpenAIProtocolClient(
        key_pool=key_pool,
        model=ANNOTATION_MODEL_NAME,
        base_url=ANNOTATION_API_BASE_URL,
        timeout_seconds=timeout_seconds,
        rate_limiter=rate_limiter,
        enable_thinking=GENERATION_CONFIG["enable_thinking"],
        thinking_mode=GENERATION_CONFIG["thinking_mode"],
        json_mode=GENERATION_CONFIG["response_format"] == "json_object",
    )
    lock = threading.Lock()

    def save():
        with lock:
            atomic_write_json(shard_path, {"results": results})

    # Resume re-attempts failed items; only completed audits are final.
    todo = [it for it in items
            if results.get(it["item_id"], {}).get("status") != "completed"]
    print(f"[ai-review shard {shard_id}] {len(todo)}/{len(items)} items to do", flush=True)
    for done, item in enumerate(todo, 1):
        try:
            entry = audit_item(client, item, data_root)
        except Exception as exc:  # noqa: BLE001
            entry = {"status": "failed", "error": str(exc)[:200]}
        results[item["item_id"]] = entry
        print(
            f"[ai-review shard {shard_id} {done}/{len(todo)}] {item['item_id']}: "
            f"{entry.get('verdict', entry.get('status'))} {entry.get('query', '')[:60]}",
            flush=True,
        )
        if done % 5 == 0 or done == len(todo):
            save()
    save()
    return results


def _plan_or_exit(args) -> dict:
    assembly_path = Path(args.assembly)
    if not assembly_path.is_file():
        raise SystemExit(f"assembly manifest not found: {assembly_path}")
    plan = build_plan(assembly_path, Path(args.data_root), Path(args.review_root))
    if not plan["items"] and not args.preflight_only:
        raise SystemExit("nothing to audit: every record already carries a human verdict")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", required=True, help="assembly.json manifest to audit")
    parser.add_argument("--data-root", default=PROJECT_ROOT / "data")
    parser.add_argument("--review-root", default=PROJECT_ROOT / "outputs" / "review",
                        help="review stores; items with a human verdict are skipped")
    parser.add_argument("--output-root", default=PROJECT_ROOT / "outputs" / "ai_review")
    parser.add_argument("--num-shards", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None, help="cap items (smoke)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="build the plan and validate images; zero API calls")
    args = parser.parse_args(argv)

    plan = _plan_or_exit(args)
    items = plan["items"]
    if args.limit:
        items = items[: args.limit]
    shards: list[list[dict]] = [[] for _ in range(max(1, args.num_shards))]
    for i, item in enumerate(items):
        shards[i % args.num_shards].append(item)
    report = {
        "protocol": AI_REVIEW_PROTOCOL,
        "prompt_hash": prompt_hash(),
        "run_tag": plan["run_tag"],
        "records_total": plan["records_total"],
        "skipped_human": plan["skipped_human"],
        "items_todo": len(items),
        "num_shards": sum(1 for s in shards if s),
        "api_calls_estimate": len(items),
    }

    if args.preflight_only:
        # Validate every referenced image exists; render one item in memory.
        missing = [it["item_id"] for it in items
                   if not (Path(args.data_root) / it["visible"]).is_file()]
        if items:
            sample = items[0]
            _load_plain(Path(args.data_root), sample["visible"])
            build_marked_annotation_view(
                _load_plain(Path(args.data_root), sample["visible"]), sample["bbox"]
            )
        report["missing_images"] = len(missing)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        if missing:
            print(f"first missing: {missing[:5]}", file=sys.stderr)
            return 2
        print("preflight OK: zero API calls", flush=True)
        return 0

    keys = load_api_keys()
    if not keys:
        raise SystemExit("No API keys found in keys/api_keys.txt")
    key_pool = APIKeyPool(keys, notify=print,
                          persist_retire=lambda i, r: comment_out_key(DEFAULT_KEY_FILE, i, r))
    rate_limiter = SlidingWindowRateLimiter(
        requests_per_minute=ANNOTATION_REQUESTS_PER_MINUTE,
        tokens_per_minute=ANNOTATION_TOKENS_PER_MINUTE,
        estimated_tokens_per_request=2500,
    )

    calls_total = {"n": 0}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, max(1, len(items)))) as pool:
        futures = []
        for shard_id, shard_items in enumerate(shards):
            if not shard_items:
                continue
            futures.append(pool.submit(
                review_shard, shard_id, shard_items, Path(args.data_root),
                Path(args.output_root), plan["run_tag"], key_pool, rate_limiter,
                timeout_seconds=args.timeout_seconds,
            ))
        for fut in as_completed(futures):
            payload = fut.result()
            with lock:
                calls_total["n"] += sum(v.get("calls", 0) for v in payload.values())
            print(f"[main] shard done ({calls_total['n']} calls so far)", flush=True)

    merged: dict = {}
    for path in sorted((Path(args.output_root) / plan["run_tag"] / "shards").glob("shard_*.json")):
        merged.update(load_json(path).get("results", {}))
    completed = sum(1 for v in merged.values() if v.get("status") == "completed")
    verdicts = {v: sum(1 for x in merged.values() if x.get("verdict") == v)
                for v in VALID_VERDICTS}
    report.update({
        "items_completed": completed,
        "verdicts": verdicts,
        "api_calls": calls_total["n"],
    })
    atomic_write_json(Path(args.output_root) / plan["run_tag"] / "ai_review.json", {
        "metadata": report,
        "results": merged,
    })
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
