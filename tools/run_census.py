"""Run the census protocol over annotation source frames.

One enumeration call per frame: the teacher names the red-boxed category,
counts its instances, then either lists every instance of it (two or more) or
lists other confident objects instead (exactly one). Code validates the
envelope -- schema, sequential indices, self-reported count -- and normalizes
ordering, duplicates and degenerate boxes.

Attributes are read for the selected frames of each sequence. No queries are
generated here.
"""

from __future__ import annotations

import argparse
import time
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from aicomp_grounding.annotation.imaging import build_marked_annotation_view, jpeg_data_url
from aicomp_grounding.annotation.client import client_for, resolve_api_key, validate_concurrency
from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.annotation.census import (
    ATTR_PROMPT_HASH,
    FINDALL_PROMPT_HASH,
    attr_messages,
    findall_messages,
    parse_attr_response,
    parse_findall_response,
    select_frames,
    trusted_objects,
)
from aicomp_grounding.annotation.depth import frame_depth_facts, load_depth_millimeters, raw_depth_path
from aicomp_grounding.annotation.config import ANNOTATION_API_BASE_URL, ANNOTATION_MODEL_LICENSE, ANNOTATION_MODEL_NAME, ANNOTATION_MODEL_REVISION, ANNOTATION_MODEL_WEIGHTS_URL, ANNOTATION_PROVIDER
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.contract import source_fingerprint
from aicomp_grounding.sharding import group_keys_by_scene, select_scene_ids, shard_scene_ids
from aicomp_grounding.annotation.source import image_fingerprint, load_annotation_source, preparation_fingerprint
from aicomp_grounding.paths import output_dir

CENSUS_PROTOCOL_VERSION = 4
FINDALL_MAX_TOKENS = 8192
ATTR_MAX_TOKENS = 1024
#: Per-stage decoding.  ``temperature: None`` and ``do_sample: None`` send
#: nothing, so the provider applies its own default; ``do_sample: False`` is
#: greedy.
GENERATION_CONFIG = {
    "enumeration": {
        "max_tokens": FINDALL_MAX_TOKENS,
        "temperature": 0.4,
        "do_sample": None,
        "thinking_mode": "enabled",
        "response_format": None,
    },
    "attr": {
        "max_tokens": ATTR_MAX_TOKENS,
        "temperature": None,
        "do_sample": False,
        "thinking_mode": "disabled",
        "response_format": None,
    },
    "image_detail": "high",
}
SELECTED_FRAMES_PER_SEQUENCE = 3


def _validate_options(split, limit_sequences, concurrency) -> str:
    normalized = split.lower().strip()
    if normalized not in {"train", "val"}:
        raise ValueError(f"Census is restricted to train/val, got {split!r}")
    if isinstance(limit_sequences, bool) or (
        limit_sequences is not None and (not isinstance(limit_sequences, int) or limit_sequences <= 0)
    ):
        raise ValueError(f"limit_sequences must be a positive integer or None, got {limit_sequences!r}")
    validate_concurrency(concurrency)
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
    index_dir: Path | None = None,
) -> dict:
    root = Path(data_root).resolve()
    dataset = load_annotation_source(root, split, index_dir=index_dir)
    identity = {
        "protocol_version": CENSUS_PROTOCOL_VERSION,
        "run_tag": run_tag,
        "split": split,
        "preparation_fingerprint": preparation_fingerprint(root, index_dir=index_dir),
        "provider": ANNOTATION_PROVIDER,
        "api_base_url": ANNOTATION_API_BASE_URL,
        "model_name": ANNOTATION_MODEL_NAME,
        "model_revision": ANNOTATION_MODEL_REVISION,
        "findall_prompt_hash": FINDALL_PROMPT_HASH,
        "attr_prompt_hash": ATTR_PROMPT_HASH,
        "generation_config": GENERATION_CONFIG,
        "seed": seed,
        "limit_sequences": limit_sequences,
        "requested_num_shards": num_shards,
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
    plan["metadata"]["image_fingerprint"] = image_fingerprint(
        root, dataset, sorted(sample_id for seq in selected_sequence_ids for sample_id in dataset if sample_id.startswith(f"{seq}_")),
        deep_verify=verify_images,
    )
    return plan


def census_preflight(*, data_root, output_root, split, limit_sequences, seed, num_shards, run_tag, resume, retry_failed, overwrite, deep_verify_images, index_dir=None) -> dict:
    plan = build_census_plan(
        data_root=data_root, split=split, limit_sequences=limit_sequences, seed=seed,
        num_shards=num_shards, run_tag=run_tag, verify_images=deep_verify_images,
        index_dir=index_dir,
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
        todo = [seq for seq in sequences
                if results.get(seq, {}).get("status") not in {"completed", "failed"}
                or (retry_failed and results.get(seq, {}).get("status") == "failed")]
        if todo:
            pending.append(shard_id)
        else:
            completed[shard_id] = payload
    return {**plan, "completed_payloads": completed, "pending_shard_ids": pending}


def _complete_once(
    client,
    build_messages,
    *,
    max_tokens: int,
    parse,
    parser_kwargs: dict,
    temperature: float | None,
    do_sample: bool | None,
) -> dict:
    """Send one request and validate the reply. Transport retries, semantics do not.

    ``client`` retries transient failures a bounded number of times, then
    raises: the run stops so the operator can look at the key, and ``--resume``
    picks up from the checkpoint. A reply that arrives but fails validation is
    recorded as a failure, not re-asked.
    """
    response = client.complete(
        messages=build_messages(),
        max_tokens=max_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    api_calls = [dict(response.record)]
    try:
        parsed = parse(response.content, **parser_kwargs)
    except ValueError as exc:
        return {
            "status": "failed",
            "attempts": 1,
            "error": str(exc),
            "api_calls": api_calls,
            "last_raw": response.content,
        }
    return {"status": "completed", "attempts": 1, "error": "", "api_calls": api_calls, **parsed}


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


def _run_findall_pass(client, marked_rgb, *, frame_ref) -> dict:
    url = jpeg_data_url(marked_rgb)
    stage = GENERATION_CONFIG["enumeration"]
    return _complete_once(
        client,
        lambda: findall_messages(url),
        max_tokens=stage["max_tokens"],
        parse=parse_findall_response,
        parser_kwargs={},
        temperature=stage["temperature"],
        do_sample=stage["do_sample"],
    )


def _aggregate_usage(results: dict) -> dict:
    totals = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for sequence in results.values():
        for frame in sequence.get("frames", {}).values():
            for key in ("findall", "attr"):
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
    api_key: str,
    resume: bool,
    retry_failed: bool,
    timeout_seconds: float,
    retry: bool,
    progress,
    index_dir: Path | None = None,
) -> dict:
    metadata = plan["metadata"]
    dataset = load_annotation_source(data_root, metadata["split"], index_dir=index_dir)
    checkpoint_path = output_root / metadata["run_id"] / "shards" / f"shard_{shard_id:02d}.json"
    results: dict = {}
    if resume and checkpoint_path.is_file():
        payload = load_json(checkpoint_path)
        if payload.get("metadata", {}).get("run_id") == metadata["run_id"]:
            results = payload.get("results", {})
    if progress is not None:
        progress.sync(results)

    def _client(stage: dict):
        return client_for(
            stage, api_key=api_key, timeout_seconds=timeout_seconds, retry=retry
        )

    enum_client = _client(GENERATION_CONFIG["enumeration"])
    attr_client = _client(GENERATION_CONFIG["attr"])
    groups = group_keys_by_scene(list(dataset), dataset)
    todo = [
        seq for seq in sequence_ids
        if results.get(seq, {}).get("status") not in {"completed", "failed"}
        or (retry_failed and results.get(seq, {}).get("status") == "failed")
    ]

    def save_checkpoint() -> None:
        atomic_write_json(
            checkpoint_path,
            {"metadata": {"run_id": metadata["run_id"], "shard_id": shard_id, "sequence_ids": list(sequence_ids)}, "results": results},
        )

    frames_since_save = 0
    try:
        for seq_offset, sequence_id in enumerate(todo, start=1):
            sample_ids = groups[sequence_id]
            frames: dict = dict(results.get(sequence_id, {}).get("frames", {}))
            sequence_result = results.setdefault(sequence_id, {})
            sequence_result.update(status="running", frames=frames)
            candidates: list[dict] = []
            for frame_offset, sample_id in enumerate(sample_ids, start=1):
                item = dataset[sample_id]
                frame_no = int(sample_id.rsplit("_", 1)[1])
                previous = frames.get(sample_id)
                if previous and previous.get("status") == "completed":
                    candidates.append(previous["candidate"])
                    continue
                if previous and previous.get("status") == "failed" and not retry_failed:
                    continue
                marked = build_marked_annotation_view(_load_plain_frame(data_root, item), item["bbox"])
                frame_started = time.monotonic()
                findall = _run_findall_pass(enum_client, marked, frame_ref=sample_id)
                frame_latency = time.monotonic() - frame_started
                completed = findall["status"] == "completed"
                frame = {
                    "findall": findall,
                    "status": "completed" if completed else "failed",
                    "error": "" if completed else findall["error"],
                }
                if completed:
                    frame["object_count"] = len(findall["objects"])
                    frame["mode"] = findall["mode"]
                    frame["candidate"] = {"sample_id": sample_id, "frame_no": frame_no, "count": len(findall["objects"])}
                    candidates.append(frame["candidate"])
                if frame["status"] == "completed":
                    depth_path = raw_depth_path(data_root, item["visible"])
                    if depth_path.is_file():
                        depth_mm = load_depth_millimeters(depth_path)
                        frame["depth"] = frame_depth_facts(depth_mm, trusted_objects(frame))
                    else:
                        frame["depth"] = {"source": "unavailable"}
                frames[sample_id] = frame
                frames_since_save += 1
                if frames_since_save >= 2:
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
                if frame["status"] == "completed":
                    detail = (
                        f"conv={findall.get('bbox_convention', '-')} "
                        f"mode={frame.get('mode', '-')} objects={frame.get('object_count', '-')} "
                        f"depth={'ok' if frame.get('depth', {}).get('source') == 'raw-uint16-mm' else '-'} "
                        f"lat={frame_latency:.1f}s"
                    )
                else:
                    detail = f"error={frame['error'][:80]}"
                print(
                    f"[shard {shard_id} seq {seq_offset}/{len(todo)} frame {frame_offset}/{len(sample_ids)}] "
                    f"{sample_id}: {frame['status']} | total={processed}/{progress.total if progress else len(plan['metadata']['selected_sequence_ids']) * 10} "
                    f"passed={passed} | {detail}",
                    flush=True,
                )
            selected_records = []
            chosen = select_frames(candidates, k=SELECTED_FRAMES_PER_SEQUENCE)
            chosen_ids = {c["sample_id"] for c in chosen}
            sequence_result["selected"] = sorted(chosen_ids)
            attr_failures = 0
            for sample_id in chosen_ids:
                frame = frames[sample_id]
                frame["selected"] = True
                agreed = trusted_objects(frame)
                indices = [obj["i"] for obj in agreed]
                item = dataset[sample_id]
                previous_attr_status = frame.get("attr", {}).get("status")
                if previous_attr_status == "completed" or (previous_attr_status == "failed" and not retry_failed):
                    attr_failures += previous_attr_status == "failed"
                    selected_records.append(frame)
                    continue
                numbered = _numbered_view(_load_plain_frame(data_root, item), agreed)
                stage = GENERATION_CONFIG["attr"]
                attr = _complete_once(
                    attr_client,
                    lambda: attr_messages(jpeg_data_url(numbered)),
                    max_tokens=stage["max_tokens"],
                    parse=parse_attr_response,
                    parser_kwargs={"indices": indices},
                    temperature=stage["temperature"],
                    do_sample=stage["do_sample"],
                )
                frame["attr"] = attr
                save_checkpoint()
                if attr["status"] != "completed":
                    attr_failures += 1
                selected_records.append(frame)
            # Sequence gate (admin 2026-09-06): only the SELECTED frames must
            # be completed. Selection draws from completed candidates, so a
            # failed non-selected frame is discarded on its own — it never
            # dooms the sequence (帧废弃不连坐序列).
            sequence_failed = not chosen_ids or any(
                frames[sid]["status"] != "completed" for sid in chosen_ids
            )
            results[sequence_id] = {
                "status": "failed" if sequence_failed else "completed",
                "frames": frames,
                "selected": sorted(chosen_ids),
                "attr_failures": attr_failures,
            }
            selected_counts = [
                frames[sid].get("object_count", frames[sid].get("agreement", {}).get("matched", 0))
                for sid in sorted(chosen_ids)
            ]
            save_checkpoint()
            print(
                f"[shard {shard_id} {seq_offset}/{len(todo)}] {sequence_id}: "
                f"{results[sequence_id]['status']} selected={sorted(chosen_ids)} objects={selected_counts}",
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
    counts = [f["object_count"] for f in completed if f.get("object_count") is not None]
    modes = [f.get("mode", "-") for f in completed]
    report = {
        "run_id": metadata["run_id"],
        "frames_total": len(frames_all),
        "frames_completed": len(completed),
        "frames_failed": len(frames_all) - len(completed),
        "enumeration_mode_distribution": {mode: modes.count(mode) for mode in sorted(set(modes))},
        "count_distribution": {},
        "sequences_ordinal_capable_ge3": 0,
        "attr_success_rate": 0.0,
        "usage": _aggregate_usage(results),
    }
    for count in sorted(set(counts)):
        report["count_distribution"][str(count)] = counts.count(count)
    capable = 0
    attr_done = attr_ok = 0
    for sequence in results.values():
        selected_counts = [
            sequence["frames"][sid].get("object_count")
            for sid in sequence.get("selected", [])
            if sequence["frames"].get(sid, {}).get("status") == "completed"
        ]
        selected_counts = [count for count in selected_counts if count is not None]
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
    index_dir: Path | None = None,
    output_root: Path = Path("outputs/census"),
    limit_sequences: int | None = 20,
    seed: int = 42,
    concurrency: int = 8,
    num_shards: int | None = None,
    run_tag: str = "",
    resume: bool = True,
    retry_failed: bool = True,
    overwrite: bool = False,
    preflight_only: bool = False,
    deep_verify_images: bool = False,
    timeout_seconds: float = 180.0,
    retry: bool = True,
    api_key: str | None = None,
) -> dict:
    split = _validate_options(split, limit_sequences, concurrency)
    output_root = output_root.resolve()
    plan = census_preflight(
        data_root=data_root, output_root=output_root, split=split,
        limit_sequences=limit_sequences, seed=seed,
        num_shards=num_shards if num_shards is not None else concurrency,
        run_tag=run_tag, resume=resume, retry_failed=retry_failed,
        overwrite=overwrite, deep_verify_images=deep_verify_images,
        index_dir=index_dir,
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
    api_key = resolve_api_key(api_key) if plan["pending_shard_ids"] else ""
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
                index_dir=index_dir,
                output_root=output_root,
                api_key=api_key,
                resume=resume,
                retry_failed=retry_failed,
                timeout_seconds=timeout_seconds,
                retry=retry,
                progress=progress,
            ))
        for future in as_completed(futures):
            payloads.append(future.result())
    report = finalize_census(plan=plan, payloads=payloads, output_root=output_root)
    print(f"Census run_id: {report['run_id']}")
    modes = report["enumeration_mode_distribution"]
    print(
        f"Frames: {report['frames_completed']}/{report['frames_total']} completed | "
        f"enumeration modes: " + ", ".join(f"{k}={v}" for k, v in modes.items())
    )
    print(f"Ordinal-capable sequences (all 3 selected frames >=3 objects): {report['sequences_ordinal_capable_ge3']}")
    print(f"Attr success: {report['attr_success_rate']:.1%}")
    print(f"API usage: {report['usage']['api_calls']} calls, {report['usage']['prompt_tokens']} input, {report['usage']['completion_tokens']} output tokens")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v5 census protocol over source frames.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=None,
                        help="source indexes (default: this repo's data/indexes; legacy data-root/indexes accepted)")
    parser.add_argument("--output-root", type=Path, default=output_dir("census"))
    parser.add_argument("--limit-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="parallel shards, each shard issues one call at a time (default 8)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="shard count; defaults to --concurrency. Part of the run fingerprint - keep stable when resuming")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--deep-verify-images", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--api-key", default=None,
                        help="key for this run; defaults to $ANNOTATION_API_KEY")
    parser.add_argument("--retry", action=argparse.BooleanOptionalAction, default=True,
                        help="retry transient failures before stopping (default: on)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_census(**vars(args))


if __name__ == "__main__":
    main()
