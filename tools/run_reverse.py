"""Reverse pass: one box per query, produced by the commercial teacher.

Same kernel as the serving side: parse -> enumerate -> sort in code -> take the
k-th, with a direct read as the retreat.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from aicomp_grounding.annotation.api import client_for, resolve_api_key, validate_concurrency
from aicomp_grounding.annotation.reverse import (
    PROMPT_HASHES,
    direct_messages,
    enumerate_messages,
    parse_direct_response,
    parse_enumeration_response,
    parse_intent_response,
    parse_messages,
    pick_kth,
)
from aicomp_grounding.annotation.views import jpeg_data_url
from aicomp_grounding.annotation.utils import (
    ANNOTATION_MODEL_NAME,
    ANNOTATION_MODEL_REVISION,
    atomic_write_json,
    stable_json_hash,
)

REVERSE_PROTOCOL_VERSION = 1

#: Per-stage decoding.  ``temperature: None`` and ``do_sample: None`` omit the
#: field so the provider applies its own default; ``False`` is greedy.
STAGE_CONFIG = {
    "parse": {
        "max_tokens": 256,
        "temperature": None,
        "do_sample": False,
        "thinking_mode": "disabled",
        "response_format": "json_object",
    },
    "enumerate": {
        "max_tokens": 8192,
        "temperature": None,
        "do_sample": None,
        "thinking_mode": "enabled",
        "response_format": None,
    },
    "direct": {
        "max_tokens": 128,
        "temperature": None,
        "do_sample": False,
        "thinking_mode": "disabled",
        "response_format": None,
    },
}

def build_reverse_plan(*, queries_path: Path, run_tag: str) -> dict:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries, dict):
        raise ValueError("The queries file must be a JSON object keyed by sample id")
    query_ids = sorted(queries)
    identity = {
        "protocol_version": REVERSE_PROTOCOL_VERSION,
        "run_tag": run_tag,
        "provider_model": ANNOTATION_MODEL_NAME,
        "model_revision": ANNOTATION_MODEL_REVISION,
        "prompt_hashes": PROMPT_HASHES,
        "stage_config": STAGE_CONFIG,
        "query_hash": stable_json_hash({key: queries[key].get("query", "") for key in query_ids}, length=32),
    }
    return {
        "queries": queries,
        "query_ids": query_ids,
        "metadata": {
            **identity,
            "run_id": f"reverse_{stable_json_hash(identity, length=16)}",
            "queries_path": str(queries_path),
        },
    }


def _complete(client, messages, stage: dict) -> tuple[str, dict]:
    response = client.complete(
        messages=messages,
        max_tokens=stage["max_tokens"],
        temperature=stage["temperature"],
        do_sample=stage["do_sample"],
    )
    return response.content, dict(response.record)


def resolve_one(clients: dict, data_root: Path, query: str, visible: str) -> dict:
    """One query -> ``{"bbox", "route", "calls", "thinking"}``."""
    calls: list[dict] = []
    thinking: list[str] = []

    text, record = _complete(clients["parse"], parse_messages(query), STAGE_CONFIG["parse"])
    calls.append(record)
    intent = parse_intent_response(text)

    if intent is None or intent["mode"] != "rank":
        route = "direct"
    else:
        route = "kernel"

    if route == "kernel":
        with Image.open(data_root / visible) as opened:
            image = opened.convert("RGB")
        url = jpeg_data_url(image)
        text, record = _complete(
            clients["enumerate"],
            enumerate_messages(url, intent["category"]),
            STAGE_CONFIG["enumerate"],
        )
        calls.append(record)
        if record.get("reasoning"):
            thinking.append(record["reasoning"])
        payload = parse_enumeration_response(text)
        if payload is None or payload["count"] == 0:
            route = "fallback-enumeration-failed"
        else:
            box = pick_kth(
                payload["boxes"],
                k=intent["k"],
                axis=intent["axis"],
                direction=intent["direction"],
            )
            if box is None:
                route = "fallback-k-out-of-range"
            else:
                return {"bbox": box, "route": "kernel", "calls": calls, "thinking": thinking}

    with Image.open(data_root / visible) as opened:
        image = opened.convert("RGB")
    text, record = _complete(
        clients["direct"], direct_messages(jpeg_data_url(image), query), STAGE_CONFIG["direct"]
    )
    calls.append(record)
    return {
        "bbox": parse_direct_response(text),
        "route": route,
        "calls": calls,
        "thinking": thinking,
    }


class _Lock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.predictions: dict = {}
        self.routes: dict = {}
        self.thinking: dict = {}
        self.calls = 0
        self.done = 0

    def record(self, key: str, outcome: dict) -> None:
        with self._lock:
            self.predictions[key] = outcome["bbox"]
            self.routes[key] = outcome["route"]
            if outcome["thinking"]:
                self.thinking[key] = outcome["thinking"][-1]
            self.calls += len(outcome["calls"])
            self.done += 1


def _load_results(path: Path) -> dict:
    """Queries already answered in an earlier run, keyed by sample id."""
    if not path.is_file():
        return {}
    done: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        done[row.pop("key")] = row
    return done


def _append_result(path: Path, key: str, outcome: dict) -> None:
    """Persist one answered query as it arrives, so a crash costs nothing."""
    row = json.dumps({"key": key, **outcome}, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(row + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def run_reverse(
    *,
    queries_path: Path,
    data_root: Path,
    output_root: Path = Path("outputs/reverse"),
    run_tag: str = "",
    concurrency: int = 8,
    timeout_seconds: float = 180.0,
    limit: int = 0,
    api_key: str | None = None,
    retry: bool = True,
    resume: bool = True,
    force: bool = False,
) -> dict:
    validate_concurrency(concurrency)
    plan = build_reverse_plan(queries_path=queries_path, run_tag=run_tag)
    run_dir = output_root.resolve() / plan["metadata"]["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    queries = plan["queries"]
    order = plan["query_ids"][:limit] if limit > 0 else plan["query_ids"]
    key = resolve_api_key(api_key)

    journal = run_dir / "results.jsonl"
    if force and journal.exists():
        journal.unlink()
    already = _load_results(journal) if resume else {}
    if already:
        print(f"[reverse] resuming: {len(already)} quer(ies) already answered", flush=True)
    todo = [sample_id for sample_id in order if sample_id not in already]

    state = _Lock()
    started = time.monotonic()

    clients = {
        name: client_for(stage, api_key=key, timeout_seconds=timeout_seconds, retry=retry)
        for name, stage in (
            ("parse", STAGE_CONFIG["parse"]),
            ("enumerate", STAGE_CONFIG["enumerate"]),
            ("direct", STAGE_CONFIG["direct"]),
        )
    }

    for sample_id, outcome in already.items():
        state.record(sample_id, outcome)

    def worker(sample_id: str) -> dict:
        entry = queries[sample_id]
        return resolve_one(clients, data_root, entry["query"], entry["visible"])

    with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(todo)))) as executor:
        futures = {
            executor.submit(worker, sample_id): sample_id for sample_id in todo
        }
        for index, future in enumerate(as_completed(futures), start=1):
            sample_id = futures[future]
            outcome = future.result()
            state.record(sample_id, outcome)
            _append_result(journal, sample_id, outcome)
            if index % 25 == 0:
                print(f"[reverse] {index}/{len(todo)}", flush=True)

    atomic_write_json(run_dir / "predictions.json", state.predictions)
    atomic_write_json(run_dir / "routes.json", state.routes)
    _write_jsonl(
        run_dir / "thinking.jsonl",
        [{"key": key, "thinking": text} for key, text in sorted(state.thinking.items())],
    )
    routes = state.routes
    metadata = {
        **plan["metadata"],
        "queries": len(todo),
        "api_calls": state.calls,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "route_counts": {
            name: sum(1 for value in routes.values() if value == name)
            for name in sorted(set(routes.values()))
        },
        "empty_predictions": sum(1 for value in state.predictions.values() if value is None),
    }
    atomic_write_json(run_dir / "metadata.json", metadata)
    print(f"[reverse] run_id={metadata['run_id']} queries={len(todo)} routes={metadata['route_counts']}")
    print(f"[reverse] wrote {run_dir}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Answer a queries file with the commercial teacher.")
    parser.add_argument("--queries", type=Path, required=True, dest="queries_path",
                        help="JSON object keyed by sample id; each value needs query and visible")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reverse"))
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--api-key", default=None,
                        help="key for this run; defaults to $ANNOTATION_API_KEY")
    parser.add_argument("--retry", action=argparse.BooleanOptionalAction, default=True,
                        help="retry transient failures before stopping (default: on)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="reuse the answers already in the run dir (default: on)")
    parser.add_argument("--force", action="store_true",
                        help="start over in an existing run dir, discarding its answers")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_reverse(**vars(args))


if __name__ == "__main__":
    main()
