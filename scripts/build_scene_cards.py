"""Build per-sequence scene cards from sampled marked RGB frames.

The old strategy had to guess whether a frame can support ordinal, spatial,
distance, or location queries. This script asks the teacher once per sampled
frame to produce a structured card, then downstream planning uses those cards
to decide which official template families are feasible.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from aicomp_grounding.annotation_state import build_marked_annotation_view, jpeg_data_url
from aicomp_grounding.api_client import (
    APIError,
    OpenAIProtocolClient,
    SlidingWindowRateLimiter,
)
from aicomp_grounding.config import (
    ANNOTATION_API_BASE_URL,
    ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST,
    ANNOTATION_MODEL_NAME,
    ANNOTATION_REQUESTS_PER_MINUTE,
    ANNOTATION_SPLITS,
    ANNOTATION_TOKENS_PER_MINUTE,
)
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.query_style import SCENE_CARD_PROMPT, parse_scene_card
from aicomp_grounding.sharding import group_keys_by_scene, select_scene_ids

SCENE_CARD_MAX_TOKENS = 2048
SCENE_CARD_TEMPERATURE = 0.0
MAX_API_CONCURRENCY = 8


def _image_part(image: Image.Image) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": jpeg_data_url(image), "detail": "high"},
    }


def build_scene_card_messages(marked_rgb: Image.Image) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a precise visual-scene analyst. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "One complete RGB scene with the tracked object enclosed by "
                        "a red rectangle."
                    ),
                },
                _image_part(marked_rgb),
                {"type": "text", "text": SCENE_CARD_PROMPT},
            ],
        },
    ]


def sample_frame_ids(
    frame_ids: list[str],
    *,
    frame_count: int,
) -> list[str]:
    frame_ids = sorted(frame_ids)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if len(frame_ids) <= frame_count:
        return frame_ids
    indexes = sorted(
        {
            round(index * (len(frame_ids) - 1) / (frame_count - 1))
            for index in range(frame_count)
        }
    )
    return [frame_ids[index] for index in indexes]


def read_scene_card(
    client: OpenAIProtocolClient,
    marked_rgb: Image.Image,
) -> dict:
    response = client.complete(
        messages=build_scene_card_messages(marked_rgb),
        max_tokens=SCENE_CARD_MAX_TOKENS,
        temperature=SCENE_CARD_TEMPERATURE,
    )
    return parse_scene_card(response.content)


def load_annotation_source(data_root: Path, split: str) -> dict:
    path = data_root / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Annotation source index not found: {path}")
    data = load_json(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{split} annotation source is empty")
    return data


def build_scene_cards(
    *,
    data_root: Path,
    index_root: Path,
    split: str,
    output_root: Path,
    seed: int,
    limit_sequences: int | None,
    frame_count: int,
    concurrency: int,
    api_key: str,
    overwrite: bool,
    timeout_seconds: float,
    requests_per_minute: int,
    tokens_per_minute: int,
    estimated_tokens_per_request: int,
) -> Path:
    dataset = load_annotation_source(index_root.resolve(), split)
    groups = group_keys_by_scene(list(dataset), dataset)
    selected = select_scene_ids(dataset, limit=limit_sequences, seed=seed)
    run_dir = output_root.resolve() / "scene_cards" / split
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cards_path = run_dir / "cards.json"

    cards: dict[str, dict] = {}
    if cards_path.is_file() and not overwrite:
        cards = load_json(cards_path)
    elif cards_path.is_file() and overwrite:
        cards_path.unlink()

    pending = [
        sequence_id
        for sequence_id in selected
        if sequence_id not in cards
    ]
    if not pending:
        return cards_path
    if not api_key:
        raise RuntimeError(
            "API_KEY is not set. Scene card generation requires Zhipu API access."
        )

    limiter = SlidingWindowRateLimiter(
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        estimated_tokens_per_request=estimated_tokens_per_request,
    )
    client = OpenAIProtocolClient(
        api_key=api_key,
        model=ANNOTATION_MODEL_NAME,
        base_url=ANNOTATION_API_BASE_URL,
        timeout_seconds=timeout_seconds,
        rate_limiter=limiter,
        enable_thinking=None,
        thinking_mode="disabled",
        json_mode=False,
    )
    print(
        f"Scene cards: split={split} sequences={len(selected)} pending={len(pending)}",
        flush=True,
    )

    def process_sequence(sequence_id: str) -> tuple[str, dict | None, str]:
        checkpoint = checkpoint_dir / f"{sequence_id}.json"
        if checkpoint.is_file():
            try:
                entry = load_json(checkpoint)
                return sequence_id, entry.get("card"), ""
            except Exception:
                checkpoint.unlink(missing_ok=True)
        frame_ids = sample_frame_ids(groups[sequence_id], frame_count=frame_count)
        sampled_cards: list[dict] = []
        for frame_id in frame_ids:
            item = dataset[frame_id]
            with Image.open(data_root / item["visible"]) as opened:
                rgb = opened.convert("RGB")
            marked = build_marked_annotation_view(rgb, item["bbox"])
            sampled_cards.append(read_scene_card(client, marked))
        card = sampled_cards[0]
        for later in sampled_cards[1:]:
            for key, value in later.items():
                if key == "stable_attributes":
                    merged = list(dict.fromkeys(card["stable_attributes"] + value))
                    card["stable_attributes"] = merged
                elif not card.get(key) and value:
                    card[key] = value
        card["sampled_frame_ids"] = frame_ids
        atomic_write_json(
            checkpoint,
            {"sequence_id": sequence_id, "card": card},
        )
        return sequence_id, card, ""

    lock = threading.Lock()
    completed = 0
    with ThreadPoolExecutor(max_workers=min(concurrency, len(pending))) as executor:
        futures = [executor.submit(process_sequence, sid) for sid in pending]
        for future in as_completed(futures):
            sequence_id, card, error = future.result()
            if card is None:
                cards[sequence_id] = {
                    "sequence_id": sequence_id,
                    "error": error or "scene card generation failed",
                }
            else:
                cards[sequence_id] = card
            with lock:
                completed += 1
                if completed % 20 == 0 or completed == len(pending):
                    print(
                        f"[scene cards] {completed}/{len(pending)} sequences processed",
                        flush=True,
                    )
    atomic_write_json(cards_path, cards)
    return cards_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(ANNOTATION_SPLITS), default="train")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--index-root", type=Path, default=Path("data/indexes"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/annotation_analysis"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-sequences", type=int)
    parser.add_argument("--frame-count", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
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
    args = parser.parse_args()
    if args.frame_count <= 0 or args.frame_count > 5:
        parser.error("--frame-count must be between 1 and 5")
    if not 1 <= args.concurrency <= MAX_API_CONCURRENCY:
        parser.error(f"--concurrency must be between 1 and {MAX_API_CONCURRENCY}")
    path = build_scene_cards(
        data_root=args.data_root,
        index_root=args.index_root,
        split=args.split,
        output_root=args.output_root,
        seed=args.seed,
        limit_sequences=args.limit_sequences,
        frame_count=args.frame_count,
        concurrency=args.concurrency,
        api_key=os.environ.get("API_KEY", "").strip(),
        overwrite=args.overwrite,
        timeout_seconds=args.timeout_seconds,
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute,
        estimated_tokens_per_request=args.estimated_tokens_per_request,
    )
    print(f"Scene cards written to {path}")


if __name__ == "__main__":
    main()
