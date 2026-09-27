from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from aicomp_grounding.annotator.bbox import normalize_bbox  # noqa: E402
from aicomp_grounding.annotator.store import ABSENT_SUFFIX, AnnotationStore  # noqa: E402

SEED_ANNOTATOR = "seed"
TEACHER_ANNOTATOR = SEED_ANNOTATOR


def build_annotation_session(
    dataset_path: Path,
    *,
    split: str = "train",
    output_path: Path | None = None,
    journal_path: Path | None = None,
    data_root: Path | None = None,
) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_data = dataset.get("data", {})
    if not isinstance(dataset_data, dict):
        dataset_data = {}

    if output_path is None:
        output_path = dataset_path

    if journal_path is None:
        journal_path = PROJECT_ROOT / "outputs" / "annotations" / "journals" / f"{split}.jsonl"

    store_dir = journal_path.parent
    store = AnnotationStore(
        store_dir,
        journal_path=journal_path,
        active_dataset_path=output_path,
        data_root=data_root or (PROJECT_ROOT / "data"),
    )

    metadata = dataset.get("metadata", {})
    provenance = metadata.get("provenance", {})
    is_seed_dataset = provenance.get("source_type") == "seed"

    items: list[dict] = []
    seeds: list[tuple[str, list[float], str]] = []
    queries_to_seed: dict[str, str] = {}
    frame_counts: dict[str, int] = {}

    for item_id in sorted(dataset_data):
        entry = dataset_data[item_id]
        bbox = entry.get("bbox")
        has_bbox = isinstance(bbox, list) and len(bbox) == 4
        query = entry.get("query", "")
        has_query = bool(query)
        status = "annotated" if (has_bbox and has_query) else "pending"

        seq = item_id.split("_")[0]
        image_path = entry.get("visible", entry.get("image", ""))
        frame_id = f"{seq}_{Path(image_path).stem}" if image_path else item_id
        if "#" in item_id:
            part = item_id.split("#", 1)[1]
            ordinal = int(part) if part.isdigit() else 1
        else:
            frame_counts[frame_id] = frame_counts.get(frame_id, 0) + 1
            ordinal = frame_counts[frame_id]

        entry_annotator = entry.get("annotator")
        if not entry_annotator and status == "annotated":
            entry_annotator = SEED_ANNOTATOR if is_seed_dataset else "human"

        items.append(
            {
                "id": item_id,
                "image": image_path,
                "query": query,
                "ordinal": ordinal,
                "frame_id": frame_id,
                "gt_bbox": [float(v) for v in bbox] if has_bbox else None,
                "category": entry.get("category", ""),
                "corpus": split,
                "status": status,
                "annotator": entry_annotator,
            }
        )

    existing_boxes = store.all_boxes()
    existing_queries = store.all_queries()

    for itm in items:
        item_id = itm["id"]
        bbox = itm.get("gt_bbox")
        query = itm.get("query")
        has_bbox = bool(bbox)
        has_query = bool(query)
        entry_annotator = itm.get("annotator")
        seed_annotator = entry_annotator or ("human" if itm.get("status") == "annotated" else SEED_ANNOTATOR)

        if has_bbox and item_id not in existing_boxes:
            seeds.append((item_id, [float(v) for v in bbox], seed_annotator))
        if has_query and item_id not in existing_queries:
            queries_to_seed[item_id] = query

    if seeds:
        store.seed_many(seeds)
    if queries_to_seed:
        store.seed_queries([(qid, qtext, None) for qid, qtext in queries_to_seed.items()])

    return {
        "name": metadata.get("run_tag", dataset_path.stem),
        "items": items,
        "stats": {
            "seeded": len(seeds),
            "frames": len({it["frame_id"] for it in items}),
            "already_seeded": len(existing_boxes),
        },
        "stores": {split: store},
        "store": store,
        "split": split,
    }


WEB_ROOT = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 1_000_000
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


class AnnotatorState:
    def __init__(
        self,
        *,
        session: dict,
        images_root: Path,
        stores: dict[str, AnnotationStore],
        corpus_of: dict[str, str],
        lock_query: bool = False,
    ) -> None:
        self.session = session
        self.lock_query = lock_query
        self.images_root = Path(images_root).resolve()
        self.items = session["items"]
        self.item_by_id = {item["id"]: item for item in self.items}
        self.image_paths = {item["image"] for item in self.items}
        self.stores = stores
        self.corpus_of = corpus_of

    def store_for(self, item_id: str) -> AnnotationStore:
        corpus = self.corpus_of.get(item_id, "")
        if corpus in self.stores:
            return self.stores[corpus]
        if "store" in self.session:
            return self.session["store"]
        return next(iter(self.stores.values()))

    def _is_human(self, annotator: object) -> bool:
        return (
            isinstance(annotator, str)
            and bool(annotator)
            and not annotator.endswith(ABSENT_SUFFIX)
            and not annotator.endswith(":todo")
            and annotator != TEACHER_ANNOTATOR
        )

    def session_payload(self) -> dict:
        payload_items = []
        for item in self.items:
            store = self.store_for(item["id"])
            meta = store.meta(item["id"]) or {}
            annotator = meta.get("annotator")
            is_absent = bool(annotator and annotator.endswith(ABSENT_SUFFIX))
            edited_query = store.get_query(item["id"])
            current_bbox = store.get(item["id"])
            if current_bbox is None and not is_absent and item.get("gt_bbox") is not None:
                current_bbox = list(item["gt_bbox"])
            if annotator is None:
                annotator = item.get("annotator")
                if annotator is None and item.get("status") == "annotated" and current_bbox is not None:
                    annotator = "human"
            elif annotator == "ai-prelabel":
                annotator = SEED_ANNOTATOR
            elif annotator == f"ai-prelabel{ABSENT_SUFFIX}":
                annotator = f"{SEED_ANNOTATOR}{ABSENT_SUFFIX}"

            payload_items.append(
                {
                    "id": item["id"],
                    "image_url": "/image?src=" + quote(item["image"]),
                    "query_en": edited_query or item["query"],
                    "query_edited": edited_query is not None,
                    "bbox": current_bbox,
                    "annotator": annotator,
                    "ordinal": item["ordinal"],
                    "frame_id": item["frame_id"],
                    "gt_bbox": item["gt_bbox"],
                    "category": item.get("category", ""),
                    "corpus": item.get("corpus", ""),
                    "status": item.get("status", "pending"),
                }
            )

        frames: dict[str, list[dict]] = {}
        for entry in payload_items:
            frames.setdefault(entry["frame_id"], []).append(entry)
        reviewed_frames = sum(
            1
            for entries in frames.values()
            if entries
            and all(
                entry["bbox"] is not None and self._is_human(entry["annotator"])
                for entry in entries
            )
        )
        return {
            "mode": "review",
            "teacher_annotator": TEACHER_ANNOTATOR,
            "seed_annotator": SEED_ANNOTATOR,
            "manifest": self.session["name"],
            "total_items": len(self.items),
            "annotated": sum(1 for e in payload_items if e["bbox"] is not None),
            "human_annotated": sum(1 for e in payload_items if self._is_human(e["annotator"])),
            "total_frames": len(frames),
            "reviewed_frames": reviewed_frames,
            "items": payload_items,
        }

    def resolve_image(self, src: str) -> Path | None:
        if src not in self.image_paths:
            return None
        candidate = Path(src)
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()

        search_bases = [
            self.images_root,
            self.images_root / "Test",
            self.images_root / "Raw",
            self.images_root / "data",
            self.images_root / "data" / "Test",
            self.images_root / "data" / "Raw",
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "data" / "Test",
            PROJECT_ROOT / "data" / "Raw",
        ]
        for base in search_bases:
            try:
                target = (base / candidate).resolve()
                if target.is_file():
                    return target
            except OSError:
                continue
        return None


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "review-server/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> AnnotatorState:
        return self.server.annotator_state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} - {fmt % args}\n")
        sys.stderr.flush()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, cache: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _read_json_body(self) -> dict:
        raw_length = self.headers.get("Content-Length") or "0"
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("invalid body size")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/api/session", "/api/progress"):
            return self._send_json(self.state.session_payload())
        if parsed.path == "/image":
            src = (parse_qs(parsed.query).get("src") or [""])[0]
            resolved = self.state.resolve_image(src)
            if resolved is None:
                sys.stderr.write(f"[WARN] image not found: src={src!r}\n")
                sys.stderr.flush()
                return self._send_json({"error": "image not found"}, 404)
            content_type = (
                STATIC_TYPES.get(resolved.suffix.lower())
                or mimetypes.guess_type(str(resolved))[0]
                or "application/octet-stream"
            )
            body = resolved.read_bytes()
            return self._send_bytes(body, content_type, cache="no-cache")
        return self._serve_static(parsed.path)

    def do_PUT(self) -> None:
        path = urlparse(self.path).path
        item_id = self._match_item_route(path, "/query")
        if item_id is not None:
            return self._handle_put_query(item_id)
        item_id = self._match_item_route(path, "/bbox")
        if item_id is None:
            return self._send_json({"error": "not found"}, 404)
        if item_id not in self.state.item_by_id:
            return self._send_json({"error": f"unknown item id: {item_id}"}, 404)
        try:
            body = self._read_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send_json({"error": str(exc)}, 400)

        raw_bbox = body.get("bbox")
        annotator = body.get("annotator")
        if not isinstance(annotator, str) or not annotator.strip() or len(annotator) > 64:
            return self._send_json(
                {"error": "annotator (non-empty string, max 64 chars) is required"},
                400,
            )
        annotator_str = annotator.strip()

        if raw_bbox is None:
            if not annotator_str.endswith(ABSENT_SUFFIX):
                annotator_str = f"{annotator_str}{ABSENT_SUFFIX}"
            self.state.store_for(item_id).delete(item_id, annotator_str)
            return self._send_json({"id": item_id, "bbox": None, "annotated": False, "absent": True})

        try:
            bbox = normalize_bbox(raw_bbox)
        except ValueError as exc:
            return self._send_json({"error": f"invalid bbox: {exc}"}, 400)

        saved = self.state.store_for(item_id).set(item_id, bbox, annotator_str)
        return self._send_json({"id": item_id, "bbox": saved, "annotated": True})

    def _handle_put_query(self, item_id: str) -> None:
        if self.state.lock_query:
            return self._send_json({"error": "query editing is locked"}, 403)
        if item_id not in self.state.item_by_id:
            return self._send_json({"error": f"unknown item id: {item_id}"}, 404)
        try:
            body = self._read_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send_json({"error": str(exc)}, 400)
        query = body.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            return self._send_json({"error": "query must be a non-empty string (max 200 chars)"}, 400)
        annotator = body.get("annotator")
        if not isinstance(annotator, str) or not annotator.strip() or len(annotator) > 64:
            return self._send_json({"error": "annotator (non-empty string, max 64 chars) is required"}, 400)
        annotator_str = annotator.strip()
        stored = self.state.store_for(item_id).set_query(item_id, query.strip(), annotator_str)
        return self._send_json({"id": item_id, "query": stored, "annotator": annotator_str})

    @staticmethod
    def _match_item_route(path: str, suffix: str) -> str | None:
        prefix = "/api/item/"
        if not path.startswith(prefix) or not path.endswith(suffix):
            return None
        middle = path[len(prefix):-len(suffix)]
        if not middle or "/" in middle:
            return None
        return unquote(middle)

    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        web_root = WEB_ROOT.resolve()
        try:
            candidate = (web_root / rel).resolve()
            candidate.relative_to(web_root)
        except (OSError, ValueError):
            return self._send_json({"error": "not found"}, 404)
        if not candidate.is_file():
            return self._send_json({"error": "frontend missing"}, 404)
        content_type = (
            STATIC_TYPES.get(candidate.suffix.lower())
            or mimetypes.guess_type(str(candidate))[0]
            or "application/octet-stream"
        )
        return self._send_bytes(candidate.read_bytes(), content_type, cache="no-cache")


def create_server(
    *,
    split: str = "train",
    output_path: str | Path | None = None,
    journal_path: str | Path | None = None,
    data_root: str | Path | None = None,
    review_root: str | Path | None = None,
    host: str = "0.0.0.0",
    port: int = 0,
    lock_query: bool = False,
) -> tuple[ThreadingHTTPServer, AnnotatorState]:
    if output_path:
        active_dataset = Path(output_path)
    else:
        primary = PROJECT_ROOT / "outputs" / "annotations" / f"{split}.json"
        fallback = PROJECT_ROOT / "outputs" / f"{split}.json"
        active_dataset = primary if primary.is_file() else fallback

    if not active_dataset.is_file():
        raise FileNotFoundError(f"Dataset file not found: {active_dataset}")

    if journal_path:
        resolved_journal = Path(journal_path)
    elif review_root:
        resolved_journal = Path(review_root) / f"{split}.jsonl"
    else:
        resolved_journal = PROJECT_ROOT / "outputs" / "annotations" / "journals" / f"{split}.jsonl"

    resolved_data_root = Path(data_root).resolve() if data_root else (PROJECT_ROOT / "data")

    session = build_annotation_session(
        active_dataset,
        split=split,
        output_path=active_dataset,
        journal_path=resolved_journal,
        data_root=resolved_data_root,
    )

    stores: dict[str, AnnotationStore] = session["stores"]
    items: list[dict] = session["items"]
    corpus_of: dict[str, str] = {it["id"]: it["corpus"] for it in items}

    server = ThreadingHTTPServer((host, port), AnnotationHandler)
    server.daemon_threads = True
    server.annotator_state = AnnotatorState(
        session=session,
        images_root=resolved_data_root,
        stores=stores,
        corpus_of=corpus_of,
        lock_query=lock_query,
    )
    return server, server.annotator_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="annotation server")
    parser.add_argument("--split", default="train", help="dataset split to annotate (default: train)")
    parser.add_argument("--output", default=None, help="override active annotations JSON path")
    parser.add_argument("--journal", default=None, help="override journal JSONL path")
    parser.add_argument("--data-root", type=Path, default=None, help="root directory containing images (default: data/)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--lock-query", action="store_true", help="disallow query edits")
    args = parser.parse_args(argv)

    try:
        server, state = create_server(
            split=args.split,
            output_path=args.output,
            journal_path=args.journal,
            data_root=args.data_root,
            host=args.host,
            port=args.port,
            lock_query=args.lock_query,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"startup failed: {exc}", file=sys.stderr)
        return 2

    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else str(host)
    print(f"session={state.session['name']} items={len(state.items)}")
    print(f"serving: http://{display_host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
