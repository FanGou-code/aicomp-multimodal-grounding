"""Generate per-frame queries from red-box RGB through the annotation API."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from aicomp_grounding.annotation_state import (
    RENDER_PROTOCOL,
    annotation_attempt_numbers,
    build_annotation_plan,
    build_annotation_shard_metadata,
    build_approved_artifact,
    build_marked_annotation_view,
    jpeg_data_url,
    merge_annotation_payloads,
    pending_frames,
    pending_sequences,
    validate_annotation_checkpoint,
)
from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.config import (
    ANNOTATION_API_BASE_URL,
    ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST,
    ANNOTATION_MODEL_LICENSE,
    ANNOTATION_MODEL_NAME,
    ANNOTATION_MODEL_REVISION,
    ANNOTATION_MODEL_WEIGHTS_URL,
    ANNOTATION_PROVIDER,
    ANNOTATION_REQUESTS_PER_MINUTE,
    ANNOTATION_SPLITS,
    ANNOTATION_TEMPERATURE,
    ANNOTATION_TOKENS_PER_MINUTE,
    PREPARATION_PROTOCOL_VERSION,
)
from aicomp_grounding.images import (
    is_trusted_image_fingerprint,
    trusted_dataset_image_fingerprint,
    verify_dataset_images,
)
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.sequence import (
    parse_frame_query_candidates,
    source_fingerprint,
)
from aicomp_grounding.sharding import group_keys_by_scene
from aicomp_grounding.api_client import (
    APIError,
    OpenAIProtocolClient,
    SlidingWindowRateLimiter,
)
from aicomp_grounding.query_style import STYLE_PROMPT_HASH, build_style_prompt

GENERATION_CONFIG = {
    "query_max_tokens": 4096,
    "temperature": ANNOTATION_TEMPERATURE,
    # Zhipu's chat-completions endpoint uses a `thinking` object, not
    # `enable_thinking`. Disable it so the verification call cannot consume
    # max_tokens on hidden reasoning and return an empty final message.
    "enable_thinking": None,
    "thinking_mode": "disabled",
    "response_format": None,
    "image_detail": "high",
}
MAX_API_CONCURRENCY = 16
PREVIEW_SEQUENCE_LIMIT = 20


def _validate_options(
    split: str,
    limit_sequences: int | None,
    concurrency: int,
    publish: bool,
) -> str:
    normalized = split.lower().strip()
    if normalized not in ANNOTATION_SPLITS:
        raise ValueError(f"Annotation is restricted to train/val, got {split!r}")
    if isinstance(limit_sequences, bool) or (
        limit_sequences is not None
        and (not isinstance(limit_sequences, int) or limit_sequences <= 0)
    ):
        raise ValueError(
            f"limit_sequences must be a positive integer or None, got {limit_sequences!r}"
        )
    if not 1 <= concurrency <= MAX_API_CONCURRENCY:
        raise ValueError(
            f"concurrency must be between 1 and {MAX_API_CONCURRENCY}, got {concurrency}"
        )
    if publish and limit_sequences is not None:
        raise ValueError("Limited annotation runs can never be published")
    return normalized


def _load_annotation_source(data_root: Path, split: str) -> dict:
    path = data_root / f"{split}.json"
    manifest_path = data_root / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Annotation source index not found: {path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Annotation requires split_manifest.json from prepare_rgbdt.py: {manifest_path}"
        )
    data = load_json(path)
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("preparation_protocol_version") != PREPARATION_PROTOCOL_VERSION
    ):
        raise ValueError("Dataset split manifest is not a completed supported preparation")
    fingerprints = manifest.get("index_fingerprints")
    counts = manifest.get("index_sample_counts")
    if not isinstance(fingerprints, dict) or fingerprints.get(split) != stable_json_hash(data):
        raise ValueError(f"{split} index does not match split_manifest.json")
    if not isinstance(counts, dict) or counts.get(split) != len(data):
        raise ValueError(f"{split} sample count does not match split_manifest.json")
    return data


def _preparation_fingerprint(data_root: Path) -> str:
    return stable_json_hash(load_json(data_root / "split_manifest.json"))


def _annotation_image_fingerprint(
    data_root: Path,
    dataset: dict,
    sample_ids: list[str],
    *,
    deep_verify: bool,
) -> str:
    if deep_verify:
        return verify_dataset_images(
            data_root,
            dataset,
            sample_ids,
            require_recorded_size=True,
        )
    return trusted_dataset_image_fingerprint(
        dataset,
        sample_ids,
        require_recorded_size=True,
    )


def prepare_annotation_plan(
    *,
    data_root: str | Path,
    split: str,
    limit_sequences: int | None,
    seed: int,
    concurrency: int,
    run_tag: str,
    verify_images: bool = False,
) -> dict:
    root = Path(data_root).resolve()
    dataset = _load_annotation_source(root, split)
    generation_config = dict(GENERATION_CONFIG)
    plan = build_annotation_plan(
        dataset,
        split=split,
        provider=ANNOTATION_PROVIDER,
        api_base_url=ANNOTATION_API_BASE_URL,
        model_name=ANNOTATION_MODEL_NAME,
        model_revision=ANNOTATION_MODEL_REVISION,
        model_weights_url=ANNOTATION_MODEL_WEIGHTS_URL,
        model_license=ANNOTATION_MODEL_LICENSE,
        prompt_hash=STYLE_PROMPT_HASH,
        render_protocol=RENDER_PROTOCOL,
        generation_config=generation_config,
        preparation_fingerprint=_preparation_fingerprint(root),
        image_fingerprint="verification-skipped",
        seed=seed,
        limit_sequences=limit_sequences,
        requested_num_shards=concurrency,
        run_tag=run_tag,
    )
    plan["metadata"]["image_fingerprint"] = _annotation_image_fingerprint(
        root,
        dataset,
        plan["selected_sample_ids"],
        deep_verify=verify_images,
    )
    return plan


def preflight_annotation_run(
    *,
    data_root: Path,
    output_root: Path,
    split: str = "train",
    limit_sequences: int | None = None,
    seed: int = 42,
    concurrency: int = 4,
    run_tag: str = "",
    resume: bool = True,
    retry_failed: bool = False,
    overwrite: bool = False,
    deep_verify_images: bool = False,
) -> dict:
    plan = prepare_annotation_plan(
        data_root=data_root,
        split=split,
        limit_sequences=limit_sequences,
        seed=seed,
        concurrency=concurrency,
        run_tag=run_tag,
        verify_images=deep_verify_images,
    )
    run_dir = output_root.resolve() / plan["metadata"]["run_id"] / split
    if overwrite and run_dir.is_dir():
        output_base = output_root.resolve()
        run_dir.resolve().relative_to(output_base)
        shutil.rmtree(run_dir)
    plan_path = run_dir / "plan.json"
    if plan_path.is_file():
        existing = load_json(plan_path)
        if existing != plan:
            raise ValueError(
                f"An incompatible annotation plan already exists at {plan_path}; change run_tag"
            )
    else:
        atomic_write_json(plan_path, plan)

    dataset = _load_annotation_source(data_root.resolve(), split)
    completed_payloads: list[dict] = []
    pending_shard_ids: list[int] = []
    for shard_id, assigned_sequence_ids in enumerate(plan["shards"]):
        checkpoint_path = run_dir / "shards" / f"shard_{shard_id:02d}.json"
        if not checkpoint_path.is_file():
            pending_shard_ids.append(shard_id)
            continue
        if not resume:
            raise FileExistsError(
                f"Checkpoint exists at {checkpoint_path}; use --resume or --overwrite"
            )
        expected_metadata = build_annotation_shard_metadata(
            plan["metadata"], shard_id, assigned_sequence_ids, dataset
        )
        payload = load_json(checkpoint_path)
        results = validate_annotation_checkpoint(
            payload,
            expected_metadata,
            assigned_sequence_ids,
            dataset,
            require_complete=False,
            label=f"annotation shard {shard_id} checkpoint",
        )
        if pending_sequences(assigned_sequence_ids, results, retry_failed=retry_failed):
            pending_shard_ids.append(shard_id)
        else:
            completed_payloads.append(payload)
    return {
        **plan,
        "completed_payloads": completed_payloads,
        "pending_shard_ids": pending_shard_ids,
    }


def _image_part(image: Image.Image) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": jpeg_data_url(image), "detail": "high"},
    }


def _frame_visual_content(marked_rgb: Image.Image) -> list[dict]:
    return [
        {
            "type": "text",
            "text": "One complete RGB scene with the intended object enclosed by a red rectangle.",
        },
        _image_part(marked_rgb),
    ]


def _messages(
    visual_content: list[dict],
    *,
    prompt: str,
    previous_error: str = "",
) -> list[dict]:
    repair = ""
    if previous_error:
        repair = (
            "\nThe previous response failed deterministic QC for this reason: "
            f"{previous_error}. Correct that failure and regenerate the complete JSON object."
        )
    return [
        {
            "role": "system",
            "content": "You are a precise visual-grounding annotator. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": [
                *visual_content,
                {"type": "text", "text": prompt + repair},
            ],
        },
    ]


def _load_plain_frame(data_root: Path, item: dict) -> Image.Image:
    with Image.open(data_root / item["visible"]) as opened:
        return opened.convert("RGB")


def _load_marked_frame(data_root: Path, item: dict) -> Image.Image:
    return build_marked_annotation_view(_load_plain_frame(data_root, item), item["bbox"])


def _annotate_frame(
    client: OpenAIProtocolClient,
    marked_rgb: Image.Image,
    *,
    previous: dict | None,
    retry_failed: bool,
    style_fields: dict | None = None,
) -> dict:
    keep_history = bool(previous and retry_failed)
    api_calls = list(previous.get("api_calls", [])) if keep_history else []
    last_error = previous.get("error", "") if previous else ""
    attempts = 0
    uncertain = False
    for run_attempt, attempt_number in enumerate(
        annotation_attempt_numbers(previous, retry_failed=retry_failed), start=1
    ):
        attempts = attempt_number
        if not style_fields or not style_fields.get("annotation_style"):
            raise ValueError(
                "Query generation requires an expanded style-plan source with annotation_style"
            )
        prompt = build_style_prompt(
            style_fields["annotation_style"],
            min_words=int(style_fields.get("annotation_min_words", 7)),
            max_words=int(style_fields.get("annotation_max_words", 13)),
            template_family=style_fields.get("annotation_style_family", "OFFICIAL"),
            fallback_style=style_fields.get(
                "annotation_fallback_style", "attribute_action"
            ),
        )
        try:
            response = client.complete(
                messages=_messages(
                    _frame_visual_content(marked_rgb),
                    prompt=prompt,
                    previous_error=last_error if run_attempt > 1 else "",
                ),
                max_tokens=GENERATION_CONFIG["query_max_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
            )
        except APIError as exc:
            last_error = f"API error: {exc}"
            continue
        api_calls.append(response.record)
        try:
            candidates = parse_frame_query_candidates(response.content)
        except ValueError as exc:
            last_error = str(exc)
            continue
        uncertain = bool(candidates["uncertain"])
        return {
            "status": "completed",
            "query": candidates["query"],
            "uncertain": uncertain,
            "attempts": attempts,
            "error": "",
            "api_calls": api_calls,
        }
    return {
        "status": "failed",
        "query": None,
        "uncertain": uncertain,
        "attempts": attempts,
        "error": last_error or "frame annotation failed deterministic QC",
        "api_calls": api_calls,
    }


class AnnotationProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self._statuses: dict[str, str] = {}
        self._lock = threading.Lock()

    def sync(self, results: dict[str, dict]) -> None:
        with self._lock:
            for result in results.values():
                for sample_id, frame in result.get("frames", {}).items():
                    self._statuses[sample_id] = frame["status"]

    def update(self, sample_id: str, status: str) -> tuple[int, int, int]:
        with self._lock:
            self._statuses[sample_id] = status
            passed = sum(value == "completed" for value in self._statuses.values())
            failed = sum(value == "failed" for value in self._statuses.values())
            return len(self._statuses), passed, failed


def annotate_shard(
    *,
    shard_id: int,
    assigned_sequence_ids: list[str],
    plan: dict,
    data_root: Path,
    output_root: Path,
    api_key: str,
    resume: bool = True,
    retry_failed: bool = False,
    timeout_seconds: float = 180.0,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    progress: AnnotationProgress | None = None,
) -> dict:
    metadata = plan["metadata"]
    dataset = _load_annotation_source(data_root, metadata["split"])
    if _preparation_fingerprint(data_root) != metadata["preparation_fingerprint"]:
        raise ValueError("Dataset preparation manifest changed after annotation preflight")
    if source_fingerprint(dataset) != metadata["source_fingerprint"]:
        raise ValueError("Annotation source changed after preflight")
    expected_metadata = build_annotation_shard_metadata(
        metadata, shard_id, assigned_sequence_ids, dataset
    )
    checkpoint_path = (
        output_root
        / metadata["run_id"]
        / metadata["split"]
        / "shards"
        / f"shard_{shard_id:02d}.json"
    )
    results: dict[str, dict] = {}
    if resume and checkpoint_path.is_file():
        results = validate_annotation_checkpoint(
            load_json(checkpoint_path),
            expected_metadata,
            assigned_sequence_ids,
            dataset,
            require_complete=False,
            label=f"annotation shard {shard_id} checkpoint",
        )
    if progress is not None:
        progress.sync(results)
    todo = pending_sequences(assigned_sequence_ids, results, retry_failed=retry_failed)
    if not todo:
        return {"metadata": expected_metadata, "results": results}

    client = OpenAIProtocolClient(
        api_key=api_key,
        model=metadata["model_name"],
        base_url=metadata["api_base_url"],
        timeout_seconds=timeout_seconds,
        rate_limiter=rate_limiter,
        enable_thinking=GENERATION_CONFIG["enable_thinking"],
        thinking_mode=GENERATION_CONFIG["thinking_mode"],
        json_mode=GENERATION_CONFIG["response_format"] == "json_object",
    )
    groups = group_keys_by_scene(list(dataset), dataset)
    preview_enabled = (
        metadata["limit_sequences"] is not None
        and metadata["limit_sequences"] <= PREVIEW_SEQUENCE_LIMIT
    )
    preview_dir = output_root / metadata["run_id"] / metadata["split"] / "preview"

    def save_checkpoint() -> None:
        atomic_write_json(checkpoint_path, {"metadata": expected_metadata, "results": results})

    try:
        for sequence_offset, sequence_id in enumerate(todo, start=1):
            sample_ids = groups[sequence_id]
            previous_sequence = results.get(sequence_id)
            frames = dict(previous_sequence.get("frames", {})) if previous_sequence else {}
            results[sequence_id] = {
                "status": "in_progress",
                "frames": frames,
            }
            save_checkpoint()

            frame_todo = pending_frames(
                sample_ids,
                frames,
                retry_failed=retry_failed,
            )
            preview_sample_id = max(
                sample_ids,
                key=lambda sample_id: (
                    (dataset[sample_id]["bbox"][2] - dataset[sample_id]["bbox"][0])
                    * (dataset[sample_id]["bbox"][3] - dataset[sample_id]["bbox"][1]),
                    sample_id,
                ),
            )
            for frame_offset, sample_id in enumerate(frame_todo, start=1):
                item = dataset[sample_id]
                plain_rgb = _load_plain_frame(data_root, item)
                marked_rgb = build_marked_annotation_view(plain_rgb, item["bbox"])
                if preview_enabled and sample_id == preview_sample_id:
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    marked_rgb.save(
                        preview_dir / f"{sequence_id}.jpg", quality=92, optimize=True
                    )
                previous_frame = frames.get(sample_id)
                frames[sample_id] = _annotate_frame(
                    client,
                    marked_rgb,
                    previous=previous_frame,
                    retry_failed=retry_failed,
                    style_fields=item,
                )
                save_checkpoint()
                frame = frames[sample_id]
                if progress is None:
                    processed = sum(
                        len(result.get("frames", {})) for result in results.values()
                    )
                    passed = sum(
                        value["status"] == "completed"
                        for result in results.values()
                        for value in result.get("frames", {}).values()
                    )
                    failed = processed - passed
                else:
                    processed, passed, failed = progress.update(sample_id, frame["status"])
                total = progress.total if progress is not None else len(plan["selected_sample_ids"])
                detail = (
                    f"query={frame['query']!r}"
                    if frame["status"] == "completed"
                    else f"error={frame['error']}"
                )
                print(
                    f"[shard {shard_id} sequence {sequence_offset}/{len(todo)} "
                    f"frame {frame_offset}/{len(frame_todo)}] {sample_id}: "
                    f"{frame['status']} | total={processed}/{total} "
                    f"passed={passed} failed={failed} | {detail}",
                    flush=True,
                )
            all_present = set(frames) == set(sample_ids)
            all_passed = all_present and all(
                frame["status"] == "completed" for frame in frames.values()
            )
            results[sequence_id]["status"] = "completed" if all_passed else "failed"
            save_checkpoint()
            print(
                f"[shard {shard_id} {sequence_offset}/{len(todo)}] "
                f"{sequence_id}: {results[sequence_id]['status']}",
                flush=True,
            )
    except Exception:
        save_checkpoint()
        raise

    payload = {"metadata": expected_metadata, "results": results}
    validate_annotation_checkpoint(
        payload,
        expected_metadata,
        assigned_sequence_ids,
        dataset,
        require_complete=True,
        label=f"annotation shard {shard_id} result",
    )
    return payload


def _aggregate_usage(results: dict[str, dict]) -> dict:
    totals = {
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_model": {},
    }
    calls: list[dict] = []
    for result in results.values():
        for frame in result.get("frames", {}).values():
            calls.extend(frame.get("api_calls", []))
    for call in calls:
        totals["api_calls"] += 1
        model_totals = totals["by_model"].setdefault(
            call["model"],
            {
                "api_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        model_totals["api_calls"] += 1
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[field] += call["usage"][field]
            model_totals[field] += call["usage"][field]
    return totals


def finalize_annotation_run(
    *,
    plan: dict,
    payloads: list[dict],
    data_root: Path,
    output_root: Path,
    publish: bool = False,
) -> dict:
    metadata = plan["metadata"]
    dataset = _load_annotation_source(data_root, metadata["split"])
    if _preparation_fingerprint(data_root) != metadata["preparation_fingerprint"]:
        raise ValueError("Dataset preparation manifest changed before finalization")
    if source_fingerprint(dataset) != metadata["source_fingerprint"]:
        raise ValueError("Annotation source changed before finalization")
    current_image_fingerprint = _annotation_image_fingerprint(
        data_root,
        dataset,
        plan["selected_sample_ids"],
        deep_verify=not is_trusted_image_fingerprint(metadata["image_fingerprint"]),
    )
    if current_image_fingerprint != metadata["image_fingerprint"]:
        raise ValueError("Annotation image references changed after preflight")
    results = merge_annotation_payloads(plan, payloads, dataset)
    failed = sorted(
        sequence_id for sequence_id, result in results.items() if result["status"] != "completed"
    )
    completed_frame_ids = {
        sample_id
        for result in results.values()
        for sample_id, frame in result.get("frames", {}).items()
        if frame["status"] == "completed"
    }
    failed_frames = sorted(set(plan["selected_sample_ids"]) - completed_frame_ids)
    uncertain_frames = sorted(
        sample_id
        for result in results.values()
        for sample_id, frame in result.get("frames", {}).items()
        if frame["status"] == "completed" and frame["uncertain"]
    )
    run_dir = output_root / metadata["run_id"] / metadata["split"]
    atomic_write_json(run_dir / "merged.json", {"metadata": metadata, "results": results})
    qc = {
        "complete": not failed and len(results) == len(metadata["selected_sequence_ids"]),
        "selected_sequences": len(metadata["selected_sequence_ids"]),
        "completed_sequences": len(results) - len(failed),
        "failed_sequences": failed,
        "selected_samples": len(plan["selected_sample_ids"]),
        "completed_samples": len(completed_frame_ids),
        "failed_frames": failed_frames,
        "uncertain_frames": uncertain_frames,
        "usage": _aggregate_usage(results),
    }
    atomic_write_json(run_dir / "qc.json", {"metadata": metadata, "qc": qc})

    approved_path = None
    if publish:
        artifact = build_approved_artifact(dataset, plan, results)
        approved_path = run_dir / "approved.json"
        if approved_path.is_file() and load_json(approved_path) != artifact:
            raise FileExistsError(
                f"A different approved artifact already exists at {approved_path}"
            )
        if not approved_path.is_file():
            atomic_write_json(approved_path, artifact)
    return {
        "run_id": metadata["run_id"],
        "split": metadata["split"],
        "qc": qc,
        "published": approved_path is not None,
        "approved_path": str(approved_path) if approved_path else None,
        "preview_path": str(run_dir / "preview") if (run_dir / "preview").is_dir() else None,
    }


def run_annotation(
    *,
    split: str = "train",
    data_root: Path = Path("data"),
    output_root: Path = Path("outputs/annotations"),
    limit_sequences: int | None = None,
    seed: int = 42,
    concurrency: int = 4,
    run_tag: str = "",
    resume: bool = True,
    retry_failed: bool = False,
    overwrite: bool = False,
    publish: bool = False,
    preflight_only: bool = False,
    deep_verify_images: bool = False,
    timeout_seconds: float = 180.0,
    requests_per_minute: int = ANNOTATION_REQUESTS_PER_MINUTE,
    tokens_per_minute: int = ANNOTATION_TOKENS_PER_MINUTE,
    estimated_tokens_per_request: int = ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST,
) -> dict:
    split = _validate_options(split, limit_sequences, concurrency, publish)
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    plan = preflight_annotation_run(
        data_root=data_root,
        output_root=output_root,
        split=split,
        limit_sequences=limit_sequences,
        seed=seed,
        concurrency=concurrency,
        run_tag=run_tag,
        resume=resume,
        retry_failed=retry_failed,
        overwrite=overwrite,
        deep_verify_images=deep_verify_images,
    )
    if preflight_only:
        print(
            f"Annotation preflight passed: {plan['metadata']['run_id']} | "
            f"pending API shards: {len(plan['pending_shard_ids'])}",
            flush=True,
        )
        return plan

    print(
        f"Starting annotation: run_id={plan['metadata']['run_id']} "
        f"split={split} sequences={len(plan['metadata']['selected_sequence_ids'])} "
        f"frames={len(plan['selected_sample_ids'])} "
        f"pending_shards={len(plan['pending_shard_ids'])}",
        flush=True,
    )

    api_key = os.environ.get("API_KEY", "").strip()
    if plan["pending_shard_ids"] and not api_key:
        raise RuntimeError(
            "API_KEY is not set. Set it in the current environment; "
            "never store it in the repository."
        )
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        estimated_tokens_per_request=estimated_tokens_per_request,
    )
    payloads = list(plan["completed_payloads"])
    progress = AnnotationProgress(len(plan["selected_sample_ids"]))
    for payload in payloads:
        progress.sync(payload["results"])
    futures = []
    with ThreadPoolExecutor(max_workers=min(concurrency, len(plan["shards"]))) as executor:
        for shard_id in plan["pending_shard_ids"]:
            futures.append(
                executor.submit(
                    annotate_shard,
                    shard_id=shard_id,
                    assigned_sequence_ids=plan["shards"][shard_id],
                    plan=plan,
                    data_root=data_root,
                    output_root=output_root,
                    api_key=api_key,
                    resume=resume,
                    retry_failed=retry_failed,
                    timeout_seconds=timeout_seconds,
                    rate_limiter=limiter,
                    progress=progress,
                )
            )
        for future in as_completed(futures):
            payloads.append(future.result())
    result = finalize_annotation_run(
        plan=plan,
        payloads=payloads,
        data_root=data_root,
        output_root=output_root,
        publish=publish,
    )
    usage = result["qc"]["usage"]
    print(f"Annotation run_id: {result['run_id']}")
    print(
        f"Sequences: {result['qc']['completed_sequences']}/"
        f"{result['qc']['selected_sequences']} completed"
    )
    print(
        f"API usage: {usage['api_calls']} calls, {usage['prompt_tokens']} input tokens, "
        f"{usage['completion_tokens']} output tokens"
    )
    for model, model_usage in sorted(usage.get("by_model", {}).items()):
        print(
            f"  {model}: {model_usage['api_calls']} calls, "
            f"{model_usage['prompt_tokens']} input tokens, "
            f"{model_usage['completion_tokens']} output tokens"
        )
    if result["approved_path"]:
        print(f"Approved artifact: {result['approved_path']}")
    if result["preview_path"]:
        print(f"Pilot previews: {result['preview_path']}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-frame visual-grounding queries from red-box RGB via annotation API."
    )
    parser.add_argument("--split", choices=sorted(ANNOTATION_SPLITS), default="train")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/annotations"))
    parser.add_argument("--limit-sequences", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--deep-verify-images", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=ANNOTATION_REQUESTS_PER_MINUTE,
    )
    parser.add_argument(
        "--tokens-per-minute",
        type=int,
        default=ANNOTATION_TOKENS_PER_MINUTE,
    )
    parser.add_argument(
        "--estimated-tokens-per-request",
        type=int,
        default=ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_annotation(**vars(args))


if __name__ == "__main__":
    main()
