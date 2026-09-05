"""Run the census protocol over annotation source frames (v5 Phase 1 pilot).

Three passes per frame: findall x2 (independent enumeration through the
red-anchored view) plus attr on the selected frames of each sequence.
Everything the teacher returns is validated in code (canary, ordering,
self-duplicates, cross-pass agreement); the run reports quality metrics and
renders ability cards for human review. No queries are assembled here —
this instrument measures the census, it does not produce training data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from foundry.annotation_views import build_marked_annotation_view, jpeg_data_url
from foundry.api_client import APIError, OpenAIProtocolClient, SlidingWindowRateLimiter
from foundry.artifacts import stable_json_hash
from foundry.census import (
    ATTR_PROMPT_HASH,
    FINDALL_PROMPT_HASH,
    attr_messages,
    draw_census_card,
    findall_messages,
    pass_agreement,
    parse_attr_response,
    parse_findall_response,
    reconcile_sequence,
    select_frames,
)
from foundry.config import (
    ANNOTATION_API_BASE_URL,
    ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST,
    ANNOTATION_MODEL_LICENSE,
    ANNOTATION_MODEL_NAME,
    ANNOTATION_MODEL_REVISION,
    ANNOTATION_MODEL_WEIGHTS_URL,
    ANNOTATION_PROVIDER,
    ANNOTATION_REQUESTS_PER_MINUTE,
    ANNOTATION_TEMPERATURE,
    ANNOTATION_TOKENS_PER_MINUTE,
    PREPARATION_PROTOCOL_VERSION,
)
from foundry.images import (
    is_trusted_image_fingerprint,
    trusted_dataset_image_fingerprint,
    verify_dataset_images,
)
from foundry.io import atomic_write_json, load_json
from foundry.keys import APIKeyPool, APIKeyPoolExhausted, load_api_keys
from foundry.sequence import source_fingerprint
from foundry.sharding import group_keys_by_scene, select_scene_ids, shard_scene_ids
from scripts.generate_queries import (
    _annotation_image_fingerprint,
    _load_annotation_source,
    _preparation_fingerprint,
)

CENSUS_PROTOCOL_VERSION = 1
FINDALL_MAX_TOKENS = 2048
ATTR_MAX_TOKENS = 1024
GENERATION_CONFIG = {
    "findall_max_tokens": FINDALL_MAX_TOKENS,
    "attr_max_tokens": ATTR_MAX_TOKENS,
    "temperature": ANNOTATION_TEMPERATURE,
    "enable_thinking": None,
    "thinking_mode": "disabled",
    "response_format": None,
    "image_detail": "high",
}
MAX_API_CONCURRENCY = 16
CARD_SEQUENCE_LIMIT = 20
SELECTED_FRAMES_PER_SEQUENCE = 3
ATTEMPTS_PER_PASS = 3


def _validate_options(split, limit_sequences, concurrency) -> str:
    normalized = split.lower().strip()
    if normalized not in {"train", "val"}:
        raise ValueError(f"Census is restricted to train/val, got {split!r}")
    if isinstance(limit_sequences, bool) or (
        limit_sequences is not None and (not isinstance(limit_sequences, int) or limit_sequences <= 0)
    ):
        raise ValueError(f"limit_sequences must be a positive integer or None, got {limit_sequences!r}")
    if not 1 <= concurrency <= MAX_API_CONCURRENCY:
        raise ValueError(f"concurrency must be between 1 and {MAX_API_CONCURRENCY}, got {concurrency}")
    return normalized


def build_census_plan(
    *,
    data_root: Path,
    split: str,
    limit_sequences: int | None,
    seed: int,
    num_shards: int,
    run_tag: str,
    verify_images: bool = False,
) -> dict:
    root = Path(data_root).resolve()
    dataset = _load_annotation_source(root, split)
    identity = {
        "protocol_version": CENSUS_PROTOCOL_VERSION,
        "run_tag": run_tag,
        "split": split,
        "preparation_fingerprint": _preparation_fingerprint(root),
        "provider": ANNOTATION_PROVIDER,
        "api_base_url": ANNOTATION_API_BASE_URL,
        "model_name": ANNOTATION_MODEL_NAME,
        "model_revision": ANNOTATION_MODEL_REVISION,
        "findall_prompt_hash": FINDALL_PROMPT_HASH,
        "attr_prompt_hash": ATTR_PROMPT_HASH,
        "generation_config": GENERATION_CONFIG,
        "seed": seed,
        "limit_sequences": limit_sequences,
    }
    run_id = f"census_{stable_json_hash(identity, length=16)}"
    selected_sequences = shard_scene_ids(
        select_scene_ids(dataset, limit=limit_sequences, seed=seed),
        dataset,
        num_shards,
    )
    selected_sequence_ids = [seq for shard in selected_sequences for seq in shard]
    metadata = {
        "protocol_version": CENSUS_PROTOCOL_VERSION,
        "run_id": run_id,
        "run_tag": run_tag,
        "split": split,
        "source_fingerprint": source_fingerprint(dataset),
        "preparation_fingerprint": identity["preparation_fingerprint"],
        "image_fingerprint": "verification-skipped",
        "provider": ANNOTATION_PROVIDER,
        "api_base_url": ANNOTATION_API_BASE_URL,
        "model_name": ANNOTATION_MODEL_NAME,
        "model_revision": ANNOTATION_MODEL_REVISION,
        "model_weights_url": ANNOTATION_MODEL_WEIGHTS_URL,
        "model_license": ANNOTATION_MODEL_LICENSE,
        "findall_prompt_hash": FINDALL_PROMPT_HASH,
        "attr_prompt_hash": ATTR_PROMPT_HASH,
        "generation_config": GENERATION_CONFIG,
        "seed": seed,
        "limit_sequences": limit_sequences,
        "requested_num_shards": num_shards,
        "effective_num_shards": len(selected_sequences),
        "selected_sequence_ids": selected_sequence_ids,
        "selected_sample_hash": stable_json_hash(
            sorted(sample_id for seq in selected_sequence_ids for sample_id in dataset if sample_id.startswith(f"{seq}_")),
            length=32,
        ),
    }
    plan = {"metadata": metadata, "shards": selected_sequences}
    plan["metadata"]["image_fingerprint"] = _annotation_image_fingerprint(
        root, dataset, sorted(sample_id for seq in selected_sequence_ids for sample_id in dataset if sample_id.startswith(f"{seq}_")),
        deep_verify=verify_images,
    )
    return plan


def census_preflight(*, data_root, output_root, split, limit_sequences, seed, num_shards, run_tag, resume, retry_failed, overwrite, deep_verify_images) -> dict:
    plan = build_census_plan(
        data_root=data_root, split=split, limit_sequences=limit_sequences, seed=seed,
        num_shards=num_shards, run_tag=run_tag, verify_images=deep_verify_images,
    )
    run_dir = Path(output_root).resolve() / plan["metadata"]["run_id"]
    if overwrite and run_dir.is_dir():
        run_dir.resolve().relative_to(Path(output_root).resolve())
        shutil.rmtree(run_dir)
    plan_path = run_dir / "plan.json"
    if plan_path.is_file():
        if load_json(plan_path) != plan:
            raise ValueError(f"An incompatible census plan already exists at {plan_path}; change run_tag")
    else:
        atomic_write_json(plan_path, plan)
    completed: dict[int, dict] = {}
    pending: list[int] = []
    for shard_id in range(plan["metadata"]["effective_num_shards"]):
        checkpoint = run_dir / "shards" / f"shard_{shard_id:02d}.json"
        if not checkpoint.is_file():
            pending.append(shard_id)
            continue
        if not resume:
            raise FileExistsError(f"Checkpoint exists at {checkpoint}; use --resume or --overwrite")
        payload = load_json(checkpoint)
        if payload.get("metadata", {}).get("run_id") != plan["metadata"]["run_id"]:
            raise ValueError(f"Census shard {shard_id} checkpoint belongs to another run")
        results = payload.get("results", {})
        sequences = plan["shards"][shard_id]
        todo = [seq for seq in sequences if results.get(seq, {}).get("status") != "completed"]
        if todo or not retry_failed:
            pending.append(shard_id) if todo else None
        if not todo:
            completed[shard_id] = payload
    return {**plan, "completed_payloads": completed, "pending_shard_ids": pending}


def _complete_with_repair(client, build_messages, *, max_tokens, previous_error, parse, parser_kwargs):
    last_error = previous_error or ""
    attempts = 0
    for attempt in range(1, ATTEMPTS_PER_PASS + 1):
        attempts = attempt
        try:
            response = client.complete(
                messages=build_messages(previous_error=last_error if attempt > 1 else ""),
                max_tokens=max_tokens,
                temperature=GENERATION_CONFIG["temperature"],
            )
        except APIKeyPoolExhausted:
            raise
        except APIError as exc:
            last_error = f"API error: {exc}"
            continue
        record = dict(response.record)
        try:
            parsed = parse(response.content, **parser_kwargs)
        except ValueError as exc:
            last_error = str(exc)
            continue
        return {"status": "completed", "attempts": attempts, "error": "", "api_calls": [record], **parsed}
    return {"status": "failed", "attempts": attempts, "error": last_error or "census pass failed validation", "api_calls": []}


def _load_plain_frame(data_root: Path, item: dict) -> Image.Image:
    with Image.open(data_root / item["visible"]) as opened:
        return opened.convert("RGB")


def _numbered_view(plain: Image.Image, objects: list[dict]) -> Image.Image:
    view = plain.convert("RGB").copy()
    draw = ImageDraw.Draw(view)
    width, height = view.size
    for item in objects:
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle((x1 * width, y1 * height, x2 * width, y2 * height), outline=(0, 160, 255), width=3)
        draw.text((x1 * width + 4, max(0, y1 * height - 14)), str(item["i"]), fill=(0, 160, 255))
    return view


def _run_findall_pass(client, marked_rgb, *, gt_bbox, pass_no, frame_ref) -> dict:
    url = jpeg_data_url(marked_rgb)
    record = _complete_with_repair(
        client,
        lambda previous_error="": findall_messages(url, previous_error=previous_error),
        max_tokens=FINDALL_MAX_TOKENS,
        previous_error="",
        parse=parse_findall_response,
        parser_kwargs={"gt_bbox": gt_bbox, "image_size": marked_rgb.size},
    )
    record["pass_no"] = pass_no
    return record


def _aggregate_usage(results: dict) -> dict:
    totals = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for sequence in results.values():
        for frame in sequence.get("frames", {}).values():
            for key in ("findall_1", "findall_2", "attr"):
                record = frame.get(key)
                if not isinstance(record, dict):
                    continue
                for call in record.get("api_calls", []):
                    totals["api_calls"] += 1
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        totals[field] += call.get("usage", {}).get(field, 0)
    return totals


def census_shard(
    *,
    shard_id: int,
    sequence_ids: list[str],
    plan: dict,
    data_root: Path,
    output_root: Path,
    key_pool: APIKeyPool,
    resume: bool,
    retry_failed: bool,
    timeout_seconds: float,
    rate_limiter,
    progress,
) -> dict:
    metadata = plan["metadata"]
    dataset = _load_annotation_source(data_root, metadata["split"])
    checkpoint_path = output_root / metadata["run_id"] / "shards" / f"shard_{shard_id:02d}.json"
    results: dict = {}
    if resume and checkpoint_path.is_file():
        payload = load_json(checkpoint_path)
        if payload.get("metadata", {}).get("run_id") == metadata["run_id"]:
            results = payload.get("results", {})
    if progress is not None:
        progress.sync(results)

    client = OpenAIProtocolClient(
        key_pool=key_pool,
        model=metadata["model_name"],
        base_url=metadata["api_base_url"],
        timeout_seconds=timeout_seconds,
        rate_limiter=rate_limiter,
        enable_thinking=GENERATION_CONFIG["enable_thinking"],
        thinking_mode=GENERATION_CONFIG["thinking_mode"],
        json_mode=GENERATION_CONFIG["response_format"] == "json_object",
    )
    groups = group_keys_by_scene(list(dataset), dataset)
    todo = [
        seq for seq in sequence_ids
        if results.get(seq, {}).get("status") != "completed" or (retry_failed and results.get(seq, {}).get("status") == "failed")
    ]

    def save_checkpoint() -> None:
        atomic_write_json(
            checkpoint_path,
            {"metadata": {"run_id": metadata["run_id"], "shard_id": shard_id, "sequence_ids": list(sequence_ids)}, "results": results},
        )

    cards_enabled = (
        metadata["limit_sequences"] is not None
        and metadata["limit_sequences"] <= CARD_SEQUENCE_LIMIT
    )
    preview_dir = output_root / metadata["run_id"] / "preview"
    frames_since_save = 0
    try:
        for seq_offset, sequence_id in enumerate(todo, start=1):
            sample_ids = groups[sequence_id]
            frames: dict = dict(results.get(sequence_id, {}).get("frames", {}))
            candidates: list[dict] = []
            for frame_offset, sample_id in enumerate(sample_ids, start=1):
                item = dataset[sample_id]
                frame_no = int(sample_id.rsplit("_", 1)[1])
                previous = frames.get(sample_id)
                if previous and previous.get("status") == "completed" and not retry_failed:
                    candidates.append(previous["candidate"])
                    continue
                marked = build_marked_annotation_view(_load_plain_frame(data_root, item), item["bbox"])
                findall_1 = _run_findall_pass(client, marked, gt_bbox=item["bbox"], pass_no=1, frame_ref=sample_id)
                findall_2 = _run_findall_pass(client, marked, gt_bbox=item["bbox"], pass_no=2, frame_ref=sample_id)
                frame = {
                    "findall_1": findall_1,
                    "findall_2": findall_2,
                    "status": "completed" if findall_1["status"] == "completed" and findall_2["status"] == "completed" else "failed",
                    "error": "",
                }
                if frame["status"] == "completed":
                    agreement = pass_agreement(findall_1["objects"], findall_2["objects"])
                    frame["agreement"] = {k: agreement[k] for k in ("matched", "count_a", "count_b", "count_agree", "jaccard")}
                    frame["candidate"] = {"sample_id": sample_id, "frame_no": frame_no, "count": agreement["matched"]}
                    candidates.append(frame["candidate"])
                else:
                    frame["error"] = findall_1["error"] or findall_2["error"]
                frames[sample_id] = frame
                frames_since_save += 1
                if frames_since_save >= 4:
                    save_checkpoint()
                    frames_since_save = 0
                if progress is None:
                    processed = sum(len(r.get("frames", {})) for r in results.values()) + len(frames)
                    passed = sum(
                        1 for r in list(results.values()) + [{"frames": frames}]
                        for f in r.get("frames", {}).values() if f.get("status") == "completed"
                    )
                else:
                    processed, passed, failed = progress.update(sample_id, frame["status"])
                detail = (
                    f"findall {findall_1.get('attempts', '-')}+{findall_2.get('attempts', '-')} "
                    f"objects {frame.get('agreement', {}).get('count_a', '-')}|{frame.get('agreement', {}).get('count_b', '-')}"
                ) if frame["status"] == "completed" else f"error={frame['error'][:80]}"
                print(
                    f"[shard {shard_id} seq {seq_offset}/{len(todo)} frame {frame_offset}/{len(sample_ids)}] "
                    f"{sample_id}: {frame['status']} | total={processed}/{progress.total if progress else len(plan['metadata']['selected_sequence_ids']) * 10} "
                    f"passed={passed} | {detail}",
                    flush=True,
                )
            selected_records = []
            chosen = select_frames(candidates, k=SELECTED_FRAMES_PER_SEQUENCE)
            chosen_ids = {c["sample_id"] for c in chosen}
            attr_failures = 0
            for sample_id in chosen_ids:
                frame = frames[sample_id]
                frame["selected"] = True
                agreed = pass_agreement(frame["findall_1"]["objects"], frame["findall_2"]["objects"])["agreed_objects"]
                indices = [obj["i"] for obj in agreed]
                item = dataset[sample_id]
                numbered = _numbered_view(_load_plain_frame(data_root, item), agreed)
                attr = _complete_with_repair(
                    client,
                    lambda previous_error="", _url=jpeg_data_url(numbered), _idx=indices: attr_messages(_url, previous_error=previous_error),
                    max_tokens=ATTR_MAX_TOKENS,
                    previous_error="",
                    parse=parse_attr_response,
                    parser_kwargs={"indices": indices},
                )
                frame["attr"] = attr
                if attr["status"] != "completed":
                    attr_failures += 1
                selected_records.append(frame)
            peers = reconcile_sequence([
                pass_agreement(f["findall_1"]["objects"], f["findall_2"]["objects"])["agreed_objects"]
                for f in selected_records
                if f["findall_1"]["status"] == "completed" and f["findall_2"]["status"] == "completed"
            ])
            if cards_enabled:
                preview_dir.mkdir(parents=True, exist_ok=True)
                for sample_id in chosen_ids:
                    frame = frames[sample_id]
                    if frame["status"] != "completed":
                        continue
                    agreed = pass_agreement(frame["findall_1"]["objects"], frame["findall_2"]["objects"])["agreed_objects"]
                    card = draw_census_card(
                        _load_plain_frame(data_root, dataset[sample_id]),
                        agreed,
                        gt_bbox=dataset[sample_id]["bbox"],
                    )
                    card.save(preview_dir / f"{sample_id}.jpg", quality=92, optimize=True)
            sequence_failed = any(f["status"] != "completed" for f in frames.values()) or not frames
            results[sequence_id] = {
                "status": "failed" if sequence_failed else "completed",
                "frames": frames,
                "selected": sorted(chosen_ids),
                "peer_count": len(peers),
                "attr_failures": attr_failures,
            }
            save_checkpoint()
            print(
                f"[shard {shard_id} {seq_offset}/{len(todo)}] {sequence_id}: "
                f"{results[sequence_id]['status']} selected={sorted(chosen_ids)} peers={len(peers)}",
                flush=True,
            )
    except Exception:
        save_checkpoint()
        raise

    payload = {"metadata": {"run_id": metadata["run_id"], "shard_id": shard_id, "sequence_ids": list(sequence_ids)}, "results": results}
    return payload


class CensusProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self._statuses: dict[str, str] = {}
        self._lock = threading.Lock()

    def sync(self, results: dict) -> None:
        with self._lock:
            for sequence in results.values():
                for sample_id, frame in sequence.get("frames", {}).items():
                    self._statuses[sample_id] = frame.get("status", "")

    def update(self, sample_id: str, status: str) -> tuple[int, int, int]:
        with self._lock:
            self._statuses[sample_id] = status
            passed = sum(value == "completed" for value in self._statuses.values())
            failed = sum(value == "failed" for value in self._statuses.values())
            return len(self._statuses), passed, failed


def finalize_census(*, plan, payloads, output_root) -> dict:
    metadata = plan["metadata"]
    run_dir = output_root / metadata["run_id"]
    results: dict = {}
    for payload in payloads:
        results.update(payload["results"])
    atomic_write_json(run_dir / "merged.json", {"metadata": metadata, "results": results})

    frames_all = [f for seq in results.values() for f in seq.get("frames", {}).values()]
    completed = [f for f in frames_all if f.get("status") == "completed"]
    agreements = [f["agreement"] for f in completed]
    report = {
        "run_id": metadata["run_id"],
        "frames_total": len(frames_all),
        "frames_completed": len(completed),
        "frames_failed": len(frames_all) - len(completed),
        "count_agree_rate": (sum(1 for a in agreements if a["count_agree"]) / len(agreements)) if agreements else 0.0,
        "mean_jaccard": (sum(a["jaccard"] for a in agreements) / len(agreements)) if agreements else 0.0,
        "count_distribution": {},
        "sequences_ordinal_capable_ge3": 0,
        "attr_success_rate": 0.0,
        "usage": _aggregate_usage(results),
    }
    counts = [a["matched"] for a in agreements]
    for count in sorted(set(counts)):
        report["count_distribution"][str(count)] = counts.count(count)
    capable = 0
    attr_done = attr_ok = 0
    for sequence in results.values():
        selected_counts = [
            sequence["frames"][sid]["agreement"]["matched"]
            for sid in sequence.get("selected", [])
            if sequence["frames"].get(sid, {}).get("status") == "completed"
        ]
        if selected_counts and min(selected_counts) >= 3:
            capable += 1
        for sid in sequence.get("selected", []):
            frame = sequence["frames"].get(sid, {})
            if frame.get("status") != "completed" or "attr" not in frame:
                continue
            attr_done += 1
            attr_ok += frame["attr"]["status"] == "completed"
    report["sequences_ordinal_capable_ge3"] = capable
    report["attr_success_rate"] = attr_ok / attr_done if attr_done else 0.0
    atomic_write_json(run_dir / "report.json", report)
    return report


def run_census(
    *,
    split: str = "train",
    data_root: Path | None = None,
    output_root: Path = Path("outputs/census"),
    limit_sequences: int | None = 20,
    seed: int = 42,
    concurrency: int = 8,
    run_tag: str = "",
    resume: bool = True,
    retry_failed: bool = True,
    overwrite: bool = False,
    preflight_only: bool = False,
    deep_verify_images: bool = False,
    timeout_seconds: float = 180.0,
    requests_per_minute: int = ANNOTATION_REQUESTS_PER_MINUTE,
    tokens_per_minute: int = ANNOTATION_TOKENS_PER_MINUTE,
    estimated_tokens_per_request: int = ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST,
) -> dict:
    split = _validate_options(split, limit_sequences, concurrency)
    data_root = Path(data_root).resolve() if data_root is not None else Path(__file__).resolve().parents[1] / "data"
    output_root = output_root.resolve()
    plan = census_preflight(
        data_root=data_root, output_root=output_root, split=split,
        limit_sequences=limit_sequences, seed=seed, num_shards=concurrency,
        run_tag=run_tag, resume=resume, retry_failed=retry_failed,
        overwrite=overwrite, deep_verify_images=deep_verify_images,
    )
    if preflight_only:
        print(f"Census preflight passed: {plan['metadata']['run_id']} | pending shards: {len(plan['pending_shard_ids'])}", flush=True)
        return plan
    print(
        f"Starting census: run_id={plan['metadata']['run_id']} split={split} "
        f"sequences={len(plan['metadata']['selected_sequence_ids'])} "
        f"pending_shards={len(plan['pending_shard_ids'])}",
        flush=True,
    )
    keys = load_api_keys()
    if plan["pending_shard_ids"] and not keys:
        raise RuntimeError("No API keys found. Write one key per line into keys/api_keys.txt; never store keys in the repository history.")
    key_pool = None
    if keys:
        key_pool = APIKeyPool(keys, notify=print)
        print(f"API keys: {key_pool.size} loaded ({key_pool.describe()})", flush=True)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        estimated_tokens_per_request=estimated_tokens_per_request,
    )
    payloads = list(plan["completed_payloads"].values())
    total_frames = sum(len(seq_ids) * 10 for seq_ids in [plan["metadata"]["selected_sequence_ids"]])
    progress = CensusProgress(total_frames)
    for payload in payloads:
        progress.sync(payload["results"])
    futures = []
    with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(plan["pending_shard_ids"])))) as executor:
        for shard_id in plan["pending_shard_ids"]:
            futures.append(executor.submit(
                census_shard,
                shard_id=shard_id,
                sequence_ids=plan["shards"][shard_id],
                plan=plan,
                data_root=data_root,
                output_root=output_root,
                key_pool=key_pool,
                resume=resume,
                retry_failed=retry_failed,
                timeout_seconds=timeout_seconds,
                rate_limiter=limiter,
                progress=progress,
            ))
        for future in as_completed(futures):
            payloads.append(future.result())
    report = finalize_census(plan=plan, payloads=payloads, output_root=output_root)
    print(f"Census run_id: {report['run_id']}")
    print(
        f"Frames: {report['frames_completed']}/{report['frames_total']} completed | "
        f"count_agree {report['count_agree_rate']:.1%} | mean jaccard {report['mean_jaccard']:.3f}"
    )
    print(f"Ordinal-capable sequences (all 3 selected frames >=3 objects): {report['sequences_ordinal_capable_ge3']}")
    print(f"Attr success: {report['attr_success_rate']:.1%}")
    print(f"API usage: {report['usage']['api_calls']} calls, {report['usage']['prompt_tokens']} input, {report['usage']['completion_tokens']} output tokens")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v5 census protocol over source frames.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/census"))
    parser.add_argument("--limit-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--deep-verify-images", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--requests-per-minute", type=int, default=ANNOTATION_REQUESTS_PER_MINUTE)
    parser.add_argument("--tokens-per-minute", type=int, default=ANNOTATION_TOKENS_PER_MINUTE)
    parser.add_argument("--estimated-tokens-per-request", type=int, default=ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_census(**vars(args))


if __name__ == "__main__":
    main()
