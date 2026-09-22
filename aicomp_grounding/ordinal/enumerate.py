"""Enumeration entry point: parse every query, then enumerate the rank ones twice.

One run = one model output.  Two runs per model are compared by
``resolve.reconcile_runs``; a run that does not decode cleanly fails on its own.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from aicomp_grounding.inference_core import load_inference_items
from aicomp_grounding.inference_state import fingerprint_inputs
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.models import available_models, get_adapter
from aicomp_grounding.ordinal import loader, run
from aicomp_grounding.ordinal.parse import build_parse_messages, strict_json_object
from aicomp_grounding.ordinal.resolve import parse_selection


def build_enumerate_messages(visible_image, category: str) -> list[dict]:
    """RGB-only messages: the depth and infrared images are not enumerated from."""
    system, user = loader.load_prompt(loader.ENUMERATE)
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": visible_image},
                {"type": "text", "text": user.replace("{category}", category)},
            ],
        },
    ]


def decode_run(text: object) -> dict | None:
    """Return ``{"count", "instances"}`` from one reply, or None when malformed.

    Only the envelope is checked here; each instance's box and confidence are
    validated by ``resolve.coerce_instances`` so that validation lives in one
    place.  The count-vs-listed comparison is the truncation gate.
    """
    payload = strict_json_object(text)
    if payload is None or set(payload) != {"count", "instances"}:
        return None
    count = payload["count"]
    if isinstance(count, bool) or not isinstance(count, int):
        return None
    if not isinstance(payload["instances"], list):
        return None
    return payload


def _image_path(relative: str, data_dir: Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else (data_dir / path).resolve()


def _visible_image(item: dict, data_dir: Path, cache: OrderedDict):
    """Decode the visible image once per path; items arrive in path order."""
    from PIL import Image

    key = str(_image_path(item["visible"], data_dir))
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    with Image.open(key) as opened:
        image = opened.convert("RGB")
    cache[key] = image
    if len(cache) > run.SCENE_CACHE_MAX:
        cache.popitem(last=False)
    return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="budget for one enumeration reply (longer than a grounding box)",
    )
    parser.add_argument("--parse-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="enumeration sampling temperature; must be > 0 or the two runs cannot differ",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/enum"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-save", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0: identical runs cannot disagree")
    items, _approved = load_inference_items(args.test_json, limit=args.limit)
    keys = [item["key"] for item in items]
    dataset = {item["key"]: item for item in items}
    adapter = get_adapter(args.model)

    def metadata(stats: dict | None = None) -> dict:
        return run.build_enum_metadata(
            model=adapter.model_name,
            model_revision=adapter.model_revision,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            run_tag=args.run_tag,
            limit=args.limit,
            selected_keys=keys,
            input_fingerprint=fingerprint_inputs(dataset, keys),
            stats=stats,
        )

    run_dir = args.output_dir / metadata()["run_id"]
    parse_path = run_dir / "parse.json"
    instances_path = run_dir / "instances.json"
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run directory already exists: {run_dir} (use --resume or a new --run-tag)")
    parse_out = load_json(parse_path) if args.resume and parse_path.is_file() else {}
    instances_out = load_json(instances_path) if args.resume and instances_path.is_file() else {}

    pending = sorted(
        (item for item in items if item["key"] not in parse_out),
        key=lambda item: (item.get("visible", item["key"]), item["key"]),
    )
    stats = {
        "parsed": len(parse_out), "rank": len(instances_out), "parse_failed": 0,
        "enum_runs": 0, "enum_decode_failed": 0,
    }
    print(f"[ordinal-enum] run_id={run_dir.name} model={args.model} pending={len(pending)}/{len(items)}")

    if pending:
        adapter.load(device=args.device, lora_path=None, model_path=args.model_path)
        cache: OrderedDict = OrderedDict()
        for index, item in enumerate(pending, start=1):
            key = item["key"]
            raw = adapter.generate_messages(
                [build_parse_messages(item["query"])],
                max_new_tokens=args.parse_max_new_tokens,
                temperature=0.0,
            )[0]
            intent = strict_json_object(raw)
            parse_out[key] = {"query": item["query"], "raw": raw, "intent": intent}
            stats["parsed"] += 1

            selection = parse_selection(intent)
            if selection is None:
                stats["parse_failed"] += 1
            elif selection.mode == "rank":
                stats["rank"] += 1
                image = _visible_image(item, args.data_dir, cache)
                runs = []
                for _ in range(run.ENUM_RUNS):
                    text = adapter.generate_messages(
                        [build_enumerate_messages(image, selection.category)],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )[0]
                    payload = decode_run(text)
                    if payload is None:
                        stats["enum_decode_failed"] += 1
                    runs.append({"raw": text, "payload": payload})
                    stats["enum_runs"] += 1
                instances_out[key] = {"category": selection.category, "runs": runs}

            if index % args.batch_save == 0:
                atomic_write_json(parse_path, parse_out)
                atomic_write_json(instances_path, instances_out)
                print(f"[ordinal-enum] {index}/{len(pending)} ranked={stats['rank']}")

        atomic_write_json(parse_path, parse_out)
        atomic_write_json(instances_path, instances_out)

    atomic_write_json(run_dir / "metadata.json", metadata(stats))
    print(
        f"[ordinal-enum] done: parsed={stats['parsed']} rank={stats['rank']} "
        f"parse_failed={stats['parse_failed']} enum_runs={stats['enum_runs']} "
        f"decode_failed={stats['enum_decode_failed']}"
    )
    print(f"[ordinal-enum] wrote {run_dir}")


if __name__ == "__main__":
    main()
