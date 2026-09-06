#!/usr/bin/env python3
"""Enumeration pass (v3.1): uncapped re-enumeration of cap-saturated frames.

The census findall caps at 6 objects; frames that hit the cap likely truncated
their scene, which makes bare ordinals ("the fifth person") count an
enumerated subset instead of the visible scene. This pass re-enumerates every
capped frame twice with no object cap, cross-passed like the main census, and
writes `enumeration.json` next to the census merged.json:

    {frame_id: {"n_pass1": int, "n_pass2": int, "agree": bool,
                "counts": {head: agreed_count}, "objects": [...],
                "canary": bool, "calls": int}}

Consumers: ordinal re-ranking (scene-true counts), color arbitration,
the disambiguation pass. Trigger and consumption rules are frozen in
spec/census_protocol.md (v3.1). Resumable: rerunning skips finished frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
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
from foundry.census import (  # noqa: E402
    pass_agreement,
    parse_findall_response,
    findall_messages,
)
from foundry.config import (  # noqa: E402
    ANNOTATION_API_BASE_URL,
    ANNOTATION_MODEL_LICENSE,
    ANNOTATION_MODEL_NAME,
    ANNOTATION_MODEL_REVISION,
    ANNOTATION_MODEL_WEIGHTS_URL,
    ANNOTATION_PROVIDER,
    ANNOTATION_REQUESTS_PER_MINUTE,
    ANNOTATION_TEMPERATURE,
    ANNOTATION_TOKENS_PER_MINUTE,
)
from scripts.run_census import GENERATION_CONFIG  # noqa: E402
from foundry.io import atomic_write_json, load_json  # noqa: E402
from foundry.keys import (  # noqa: E402
    APIKeyPool,
    DEFAULT_KEY_FILE,
    comment_out_key,
    load_api_keys,
)

CENSUS_PROTOCOL_VERSION = "3.1"
ENUMERATION_PROMPT = """The red rectangle marks one object in the scene.
Task: list EVERY object of the red-boxed category that is visible in this image — including small, distant, blurry, and partially occluded instances. Do NOT stop at any number; count all visible instances of that category.
Also include other clearly identifiable objects only if slots remain after all same-category instances are listed.
Order everything from left to right. You must include the object inside the red rectangle.
For each object give a short common category name and its bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.
Output JSON only:
{"objects": [{"i": 1, "category": "<category name>", "bbox": [x1, y1, x2, y2]}, ...]}"""

FINDALL_MAX_TOKENS = 3072
ATTEMPTS_PER_PASS = 3
ENUM_HASH = None  # filled after module import


def _enum_hash() -> str:
    import hashlib

    return hashlib.sha256(ENUMERATION_PROMPT.encode("utf-8")).hexdigest()


def _load_plain(data_root: Path, item: dict):
    path = data_root / item["visible"]
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def _parse_response(text: str, gt_bbox: list[float]) -> dict:
    return parse_findall_response(text, gt_bbox=gt_bbox)


def _enum_messages(marked_jpeg_url: str, *, previous_error: str = "") -> list[dict]:
    repair = ""
    if previous_error:
        repair = (
            "\nThe previous response failed deterministic validation for this reason: "
            f"{previous_error}. Correct that failure and regenerate the complete JSON object."
        )
    return [
        {"role": "system", "content": "You are a precise visual-grounding enumerator. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "One complete RGB scene; the red rectangle marks the reference object."},
                {"type": "image_url", "image_url": {"url": marked_jpeg_url, "detail": "high"}},
                {"type": "text", "text": ENUMERATION_PROMPT + repair},
            ],
        },
    ]


def _complete_with_repair(client, marked, gt_bbox, frame_ref, key):
    last_error = ""
    api_calls = []
    last_raw = ""
    for attempt in range(1, ATTEMPTS_PER_PASS + 1):
        try:
            response = client.complete(
                messages=_enum_messages(
                    jpeg_data_url(marked), previous_error=last_error if attempt > 1 else ""
                ),
                max_tokens=FINDALL_MAX_TOKENS,
                temperature=ANNOTATION_TEMPERATURE,
            )
            record = _parse_response(response.content, gt_bbox)
            api_calls.append(response.record.get("usage", {}))
            return {
                "status": "completed",
                "attempts": attempt,
                "objects": record["objects"],
                "bbox_convention": record["bbox_convention"],
                "api_calls": api_calls,
            }
        except APIError as exc:
            last_error = str(exc)[:200]
            api_calls.append({"error": last_error})
            last_raw = last_error
        except ValueError as exc:
            last_error = str(exc)[:200]
            api_calls.append({"error": last_error})
            last_raw = last_error
    return {"status": "failed", "attempts": ATTEMPTS_PER_PASS, "error": last_error,
            "api_calls": api_calls}


def enumeration_shard(
    shard_id: int,
    frame_ids: list[str],
    frames_meta: dict,
    data_root: Path,
    output_root: Path,
    run_id: str,
    key_pool,
    rate_limiter,
    *,
    timeout_seconds: float,
    concurrency: int,
) -> dict:
    results: dict = {}
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / "enum_shards" / f"shard_{shard_id:02d}.json"
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

    todo = [fid for fid in frame_ids if fid not in results]
    print(f"[enum shard {shard_id}] {len(todo)}/{len(frame_ids)} frames to do", flush=True)
    for done, fid in enumerate(todo, 1):
        meta = frames_meta[fid]
        item = meta["item"]
        try:
            plain = _load_plain(data_root, item)
            marked = build_marked_annotation_view(plain, item["bbox"])
            p1 = _complete_with_repair(client, marked, item["bbox"], fid, "pass1")
            p2 = _complete_with_repair(client, marked, item["bbox"], fid, "pass2")
            entry: dict = {"calls": p1.get("attempts", 0) + p2.get("attempts", 0)}
            if p1["status"] == "completed" and p2["status"] == "completed":
                agreement = pass_agreement(p1["objects"], p2["objects"])
                counts: dict = {}
                for obj in agreement["agreed_objects"]:
                    head = obj["category"].split()[-1]
                    counts[head] = counts.get(head, 0) + 1
                canary = any(
                    _iou_ge(obj["bbox"], item["bbox"]) for obj in agreement["agreed_objects"]
                )
                entry.update({
                    "n_pass1": len(p1["objects"]),
                    "n_pass2": len(p2["objects"]),
                    "agree": agreement["matched"],
                    "counts": counts,
                    "objects": agreement["agreed_objects"],
                    "canary": canary,
                    "status": "completed",
                })
            else:
                entry.update({
                    "status": "failed",
                    "error": p1.get("error") or p2.get("error") or "",
                })
            results[fid] = entry
            print(
                f"[enum shard {shard_id} {done}/{len(todo)}] {fid}: {entry['status']} "
                f"n={entry.get('n_pass1', '-')}|{entry.get('n_pass2', '-')} "
                f"agree={entry.get('agree', '-')} "
                f"counts={entry.get('counts', {})}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            results[fid] = {"status": "failed", "error": str(exc)[:200]}
            print(f"[enum shard {shard_id}] {fid}: exception {str(exc)[:80]}", flush=True)
        if done % 3 == 0 or done == len(todo):
            save()
    save()
    return results


def _iou_ge(a, b, thresh=0.5):
    from foundry.bbox import compute_iou

    return compute_iou(a, b) >= thresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-run", required=True, help="census run dir with merged.json")
    parser.add_argument("--data-root", default=PROJECT_ROOT / "data")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None, help="cap frames (smoke)")
    args = parser.parse_args()

    census_dir = Path(args.census_run)
    merged = load_json(census_dir / "merged.json")
    metadata = merged.get("metadata", {})
    run_id = str(metadata.get("run_id") or census_dir.name)
    split = metadata.get("split", "train")
    index = load_json(Path(args.data_root) / "indexes" / f"{split}.json")

    # Trigger rule (frozen): frames whose main enumeration hit the cap.
    from foundry.census import pass_agreement as _pa

    capped_frames: dict[str, dict] = {}
    for seq_id, seq in merged.get("results", {}).items():
        if seq.get("status") != "completed":
            continue
        for fid, frame in seq.get("frames", {}).items():
            if frame.get("status") != "completed":
                continue
            if frame.get("single_pass"):
                good = (frame["findall_1"] if frame["findall_1"]["status"] == "completed"
                        else frame["findall_2"])
                objects = good["objects"]
            else:
                objects = _pa(
                    frame["findall_1"]["objects"], frame["findall_2"]["objects"]
                )["agreed_objects"]
            if len(objects) >= 6 and fid in index:
                capped_frames[fid] = {"item": index[fid], "n_capped": len(objects)}
    if args.limit:
        capped_frames = dict(sorted(capped_frames.items())[: args.limit])
    print(f"enumeration pass: {len(capped_frames)} capped frames "
          f"(protocol {CENSUS_PROTOCOL_VERSION}, prompt {_enum_hash()[:12]})")

    keys = load_api_keys()
    if not keys:
        raise SystemExit("No API keys found in keys/api_keys.txt")
    key_pool = APIKeyPool(keys, notify=print,
                          persist_retire=lambda i, r: comment_out_key(DEFAULT_KEY_FILE, i, r))
    rate_limiter = SlidingWindowRateLimiter(
        requests_per_minute=ANNOTATION_REQUESTS_PER_MINUTE,
        tokens_per_minute=ANNOTATION_TOKENS_PER_MINUTE,
        estimated_tokens_per_request=2200,
    )

    frame_ids = sorted(capped_frames)
    shards: list[list[str]] = [[] for _ in range(args.num_shards)]
    for i, fid in enumerate(frame_ids):
        shards[i % args.num_shards].append(fid)

    output_root = census_dir.parent
    calls_total = {"n": 0}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(frame_ids))) as pool:
        futures = []
        for shard_id, frame_ids_shard in enumerate(shards):
            if not frame_ids_shard:
                continue
            futures.append(pool.submit(
                enumeration_shard, shard_id, frame_ids_shard, capped_frames,
                Path(args.data_root), output_root, run_id, key_pool, rate_limiter,
                timeout_seconds=args.timeout_seconds, concurrency=args.concurrency,
            ))
        for fut in as_completed(futures):
            payload = fut.result()
            with lock:
                calls_total["n"] += sum(v.get("calls", 0) for v in payload.values())
            print(f"[main] shard done ({calls_total['n']} calls so far)", flush=True)

    # merge
    merged_enum: dict = {}
    for path in sorted((output_root / run_id / "enum_shards").glob("shard_*.json")):
        merged_enum.update(load_json(path).get("results", {}))
    completed = sum(1 for v in merged_enum.values() if v.get("status") == "completed")
    consistent = sum(
        1 for v in merged_enum.values()
        if v.get("status") == "completed" and v.get("n_pass1") == v.get("n_pass2")
    )
    report = {
        "protocol_version": CENSUS_PROTOCOL_VERSION,
        "enumeration_prompt_hash": _enum_hash(),
        "frames_total": len(capped_frames),
        "frames_completed": completed,
        "count_consistency": round(consistent / completed, 3) if completed else 0,
        "api_calls": calls_total["n"],
    }
    atomic_write_json(census_dir / "enumeration.json", {
        "metadata": report,
        "results": merged_enum,
    })
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
