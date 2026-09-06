"""Review server backend: gt-annotator adapted to query-foundry sessions.

Same stdlib-only HTTP core and crash-safe store as gt-annotator; the session
source is a census run (teacher boxes pre-seeded for human verification)
instead of a manifest. Review mode semantics: boxes can be dragged and
resized but never deleted; a frame counts as reviewed when every one of its
objects carries a human annotation.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.review.bbox import normalize_bbox  # noqa: E402
from foundry.review.census_session import (  # noqa: E402
    TEACHER_ANNOTATOR,
    build_assembly_session,
    build_census_session,
)
from foundry.review.store import ABSENT_SUFFIX, AnnotationStore  # noqa: E402

WEB_ROOT = Path(__file__).resolve().parent / "web"
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
    """Immutable per-run context shared across request handler threads.

    ``stores`` maps corpus ("train"/"val") to its crash-safe store, so a
    combined multi-assembly session keeps each split's journal where it has
    always lived and read/write routes by the item's corpus.
    """

    def __init__(
        self,
        *,
        session: dict,
        images_root: Path,
        stores: dict[str, AnnotationStore],
        corpus_of: dict[str, str],
    ) -> None:
        self.session = session
        self.images_root = Path(images_root).resolve()
        self.items = session["items"]
        self.item_by_id = {item["id"]: item for item in self.items}
        self.image_paths = {item["image"] for item in self.items}
        self.stores = stores
        self.corpus_of = corpus_of

    def store_for(self, item_id: str) -> AnnotationStore:
        return self.stores[self.corpus_of[item_id]]

    def _is_human(self, annotator: object) -> bool:
        return (
            isinstance(annotator, str)
            and annotator
            and not annotator.endswith(ABSENT_SUFFIX)
            and annotator != TEACHER_ANNOTATOR
        )

    def session_payload(self) -> dict:
        payload_items = []
        for item in self.items:
            store = self.store_for(item["id"])
            meta = store.meta(item["id"]) or {}
            edited_query = store.get_query(item["id"])
            payload_items.append(
                {
                    "id": item["id"],
                    "image_url": "/image?src=" + quote(item["image"]),
                    "query_en": edited_query or item["query"],
                    "query_edited": edited_query is not None,
                    "bbox": self.store_for(item["id"]).get(item["id"]),
                    "annotator": meta.get("annotator"),
                    "ordinal": item["ordinal"],
                    "frame_id": item["frame_id"],
                    "gt_bbox": item["gt_bbox"],
                    "bucket": item.get("bucket", ""),
                    "category": item.get("category", ""),
                    "corpus": item.get("corpus", ""),
                    "ai_verdict": item.get("ai_verdict", ""),
                    "ai_reason": item.get("ai_reason", ""),
                    "ai_collision": item.get("ai_collision", False),
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
            "mode": "census-review",
            "manifest": self.session["name"],
            "census_run_id": self.session.get("census_run_id"),
            "total_items": len(self.items),
            "annotated": sum(1 for e in payload_items if e["bbox"] is not None),
            "human_annotated": sum(1 for e in payload_items if self._is_human(e["annotator"])),
            "total_frames": len(frames),
            "reviewed_frames": reviewed_frames,
            "items": payload_items,
        }

    def resolve_image(self, src: str) -> Path | None:
        # Membership check against session items makes traversal impossible.
        if src not in self.image_paths:
            return None
        candidate = Path(src)
        path = candidate if candidate.is_absolute() else self.images_root / candidate
        try:
            resolved = path.resolve()
        except OSError:
            return None
        if not resolved.is_file():
            return None
        return resolved


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "query-foundry-review/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> AnnotatorState:
        return self.server.annotator_state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # keep the review console quiet

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
        if parsed.path == "/api/session":
            return self._send_json(self.state.session_payload())
        if parsed.path == "/api/progress":
            return self._send_json(self.state.session_payload())
        if parsed.path == "/image":
            src = (parse_qs(parsed.query).get("src") or [""])[0]
            resolved = self.state.resolve_image(src)
            if resolved is None:
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
        try:
            bbox = normalize_bbox(body.get("bbox"))
        except ValueError as exc:
            return self._send_json({"error": f"invalid bbox: {exc}"}, 400)
        annotator = body.get("annotator")
        if not isinstance(annotator, str) or not annotator.strip() or len(annotator) > 64:
            return self._send_json(
                {"error": "annotator (non-empty string, max 64 chars) is required in review mode"},
                400,
            )
        saved = self.state.store_for(item_id).set(item_id, bbox, annotator.strip())
        return self._send_json({"id": item_id, "bbox": saved, "annotated": True})

    def _handle_put_query(self, item_id: str) -> None:
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
            return self._send_json({"error": "annotator required"}, 400)
        stored = self.state.store_for(item_id).set_query(item_id, query.strip(), annotator.strip())
        return self._send_json({"id": item_id, "query": stored, "annotator": annotator.strip()})

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
    census_run_dir: str | Path | None = None,
    data_root: str | Path,
    review_root: str | Path,
    assembly_path: str | Path | list[str | Path] | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ThreadingHTTPServer, AnnotatorState]:
    """Build the review server.

    ``assembly_path`` takes one assembly manifest (unchanged behaviour) or a
    list of them — a combined train+val session. Each split keeps its own
    crash-safe store directory, so existing review progress carries over
    untouched; writes route by the item's corpus.
    """
    if assembly_path is not None:
        paths = [assembly_path] if isinstance(assembly_path, (str, Path)) else list(assembly_path)
        sessions = [build_assembly_session(Path(p), Path(data_root), Path(review_root)) for p in paths]
    else:
        sessions = [build_census_session(Path(census_run_dir), Path(data_root), Path(review_root))]

    items: list[dict] = []
    stores: dict[str, AnnotationStore] = {}
    corpus_of: dict[str, str] = {}
    for session in sessions:
        for item in session["items"]:
            items.append(item)
            corpus_of[item["id"]] = item["corpus"]
        split = session["split"]
        if split in stores:
            raise ValueError(f"multiple assemblies declare the same split: {split}")
        stores[split] = session["store"]

    session = {
        "name": " + ".join(s["name"] for s in sessions),
        "items": items,
        "stats": {
            key: sum(s["stats"][key] for s in sessions)
            for key in ("seeded", "frames", "already_seeded")
        },
        "stores": stores,
    }
    if len(sessions) == 1:
        session["store"] = sessions[0]["store"]  # single-corpus back-compat
        session["census_run_id"] = sessions[0].get("census_run_id")

    images_root = Path(data_root).resolve()
    server = ThreadingHTTPServer((host, port), AnnotationHandler)
    server.daemon_threads = True
    server.annotator_state = AnnotatorState(
        session=session, images_root=images_root, stores=stores, corpus_of=corpus_of
    )
    return server, server.annotator_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="census review server (gt-annotator adapted)")
    parser.add_argument("--census-run", help="census run dir with merged.json (box review mode)")
    parser.add_argument("--assembly", nargs="+", help="assembly.json path(s); several = combined session")
    parser.add_argument("--data-root", default=PROJECT_ROOT / "data")
    parser.add_argument("--review-root", default=PROJECT_ROOT / "outputs" / "review")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)

    if not args.census_run and not args.assembly:
        parser.error("either --census-run or --assembly is required")
    try:
        server, state = create_server(
            census_run_dir=args.census_run,
            data_root=args.data_root,
            review_root=args.review_root,
            assembly_path=args.assembly,
            host=args.host,
            port=args.port,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"startup failed: {exc}", file=sys.stderr)
        return 2

    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else str(host)
    print(f"session={state.session['name']} items={len(state.items)} frames={state.session['stats']['frames']}")
    print(f"seeded={state.session['stats']['seeded']} already_seeded={state.session['stats']['already_seeded']}")
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
