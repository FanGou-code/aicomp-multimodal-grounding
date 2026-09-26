"""Crash-safe annotation store.

Durable state = append-only JSONL journal (one record per PUT/DELETE).
``annotations.predictions.json`` is a consolidated snapshot in the main
project's prediction-file shape (``{item_id: [x1, y1, x2, y2]}``, normalized
0-1 XYXY, annotated items only) and is rewritten atomically on every change.
Absence verdicts are journaled as ``bbox: null`` records whose ``annotator``
ends in ``:absent`` (e.g. ``ai-prelabel:absent`` pending review, ``fang0:absent``
human-confirmed) and consolidated into ``annotations.absent.json`` as
``{item_id: annotator}``. Recovery = replay the journal; a torn trailing line
from a crash is ignored.

Snapshots are always re-derived from a full journal replay at write time, so
concurrent writer processes (the review server and the apply step) never drop each
other's entries. Readers re-replay automatically whenever the journal file
changes on disk, which hot-reloads annotations written by other processes.
Query edits are journaled as bbox-less records and replay as text + annotator
updates without touching box state.


Adapted from gt-annotator (upstream gt-annotator project,
MIT License, (c) 2026 FanGou-code) - adapted for this repo's review tool."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from aicomp_grounding.io import atomic_write_json

JOURNAL_NAME = "annotations.jsonl"
QUERY_SNAPSHOT_NAME = "annotations.queries.json"
SNAPSHOT_NAME = "annotations.predictions.json"
ABSENT_SNAPSHOT_NAME = "annotations.absent.json"
ABSENT_SUFFIX = ":absent"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _is_absent_annotator(annotator: object) -> bool:
    return isinstance(annotator, str) and annotator.endswith(ABSENT_SUFFIX)


class AnnotationStore:
    def __init__(
        self,
        data_dir: str | Path,
        *,
        journal_path: str | Path | None = None,
        active_dataset_path: str | Path | None = None,
        skeleton_path: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.journal_path = Path(journal_path) if journal_path is not None else self.data_dir / JOURNAL_NAME
        self.snapshot_path = self.data_dir / SNAPSHOT_NAME
        self.queries_path = self.data_dir / QUERY_SNAPSHOT_NAME
        self.absent_path = self.data_dir / ABSENT_SNAPSHOT_NAME
        self.active_dataset_path = Path(active_dataset_path) if active_dataset_path is not None else None
        self.skeleton_path = Path(skeleton_path) if skeleton_path is not None else None
        self._skeleton_images: dict[str, str] | None = None
        self._lock = threading.Lock()
        self._state: dict[str, list[float]] = {}
        self._queries: dict[str, str] = {}
        self._meta: dict[str, dict] = {}
        self._absent: dict[str, str] = {}
        self._touched: set[str] = set()
        self._journal_sig: tuple[int, int, int] | None = None
        self._replay()


    # -- journal replay ----------------------------------------------------

    def _read_journal_state(self) -> tuple[dict, dict, dict, dict, set[str]]:
        state: dict[str, list[float]] = {}
        queries: dict[str, str] = {}
        meta: dict[str, dict] = {}
        absent: dict[str, str] = {}
        touched: set[str] = set()
        if not self.journal_path.is_file():
            return state, queries, meta, absent, touched
        with self.journal_path.open("rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue  # torn tail from a crash; journal remains truth
                if not isinstance(record, dict):
                    continue
                item_id = record.get("id")
                if not isinstance(item_id, str) or not item_id:
                    continue
                touched.add(item_id)
                annotator = record.get("annotator")
                if record.get("query"):
                    queries[item_id] = str(record["query"])
                if "bbox" not in record:
                    # Query-only edit (set_query shape): updates the text and
                    # the annotator claim; box state is untouched. The old
                    # replay treated the missing bbox as a delete and wiped
                    # the human box on every subsequent read/replay.
                    meta[item_id] = {"annotator": annotator, "ts": record.get("ts")}
                    continue
                bbox = record["bbox"]
                if bbox is None:
                    state.pop(item_id, None)
                    if _is_absent_annotator(annotator):
                        meta[item_id] = {"annotator": annotator, "ts": record.get("ts")}
                        absent[item_id] = annotator
                    else:
                        # plain delete: back to unannotated
                        meta.pop(item_id, None)
                        absent.pop(item_id, None)
                elif isinstance(bbox, list) and len(bbox) == 4:
                    try:
                        state[item_id] = [float(v) for v in bbox]
                    except (TypeError, ValueError):
                        continue
                    meta[item_id] = {"annotator": annotator, "ts": record.get("ts")}
                    absent.pop(item_id, None)
        return state, queries, meta, absent, touched

    def _journal_signature(self) -> tuple[int, int, int] | None:
        try:
            st = self.journal_path.stat()
        except OSError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _replay(self) -> None:
        signature = self._journal_signature()
        self._state, self._queries, self._meta, self._absent, self._touched = self._read_journal_state()
        # If another writer appended during this read, leave the earlier
        # signature so the next read refreshes instead of hiding the new tail.
        self._journal_sig = signature

    def _refresh_locked(self) -> None:
        """Hot-reload in-memory state if another process appended to the journal."""
        for _ in range(5):
            sig = self._journal_signature()
            if sig == self._journal_sig:
                return
            state, queries, meta, absent, touched = self._read_journal_state()
            # Concurrent writers may have appended while we replayed; only
            # commit the replay if the file is unchanged since it started,
            # otherwise the new tail would be swallowed by this signature.
            if self._journal_signature() == sig:
                self._state, self._queries, self._meta, self._absent = state, queries, meta, absent
                self._touched = touched
                self._journal_sig = sig
                return
        # Journal kept changing under us; leave state as-is, next read retries.

    # -- reads ---------------------------------------------------------------

    def get(self, item_id: str) -> list[float] | None:
        with self._lock:
            self._refresh_locked()
            bbox = self._state.get(item_id)
            return list(bbox) if bbox is not None else None

    def meta(self, item_id: str) -> dict | None:
        with self._lock:
            self._refresh_locked()
            entry = self._meta.get(item_id)
            return dict(entry) if entry is not None else None

    def all_meta(self) -> dict[str, dict]:
        with self._lock:
            self._refresh_locked()
            return {item_id: dict(entry) for item_id, entry in self._meta.items()}

    def all_boxes(self) -> dict[str, list[float]]:
        with self._lock:
            self._refresh_locked()
            return {item_id: list(bbox) for item_id, bbox in self._state.items()}

    # -- writes --------------------------------------------------------------

    def snapshot(self) -> dict:
        """Read one coherent journal state without modifying journal or snapshots."""
        with self._lock:
            self._refresh_locked()
            return {
                "boxes": {k: list(v) for k, v in self._state.items()},
                "queries": dict(self._queries),
                "meta": {k: dict(v) for k, v in self._meta.items()},
                "absent": dict(self._absent),
                "touched_ids": set(self._touched),
            }

    @contextmanager
    def _write_lock(self):
        # Review servers run on Linux/macOS. flock coordinates separate server
        # processes/instances; the Python lock protects threads on this instance.
        import fcntl
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with (self.data_dir / ".annotations.lock").open("a+b") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)


    def set(self, item_id: str, bbox: list[float], annotator: str | None = None) -> list[float]:
        bbox = [float(v) for v in bbox]
        record = {"id": item_id, "bbox": bbox, "annotator": annotator, "ts": _now()}
        with self._write_lock():
            self._append(record)
            self._write_snapshots()
        return bbox

    def seed_many(self, seeds: list[tuple[str, list[float], str]]) -> int:
        """Bulk AI-seed: one journal append per box, one snapshot write total.

        The per-record set() path re-replays the whole journal on every write
        for cross-process safety; across a full-corpus seed that is O(n²) and
        stalled the review server for minutes before its port went up. Batch
        appends keep the journal format identical (one record per box, same
        replay semantics); excluding human-annotated items stays the caller's
        job, decided against a single all_meta() snapshot.
        """
        if not seeds:
            return 0
        records = [
            {"id": item_id, "bbox": [float(v) for v in bbox], "annotator": annotator, "ts": _now()}
            for item_id, bbox, annotator in seeds
        ]
        with self._write_lock():
            self._append_many(records)
            self._write_snapshots()
        return len(seeds)


    def set_query(self, item_id: str, query: str, annotator: str | None = None) -> str:
        """Record a human-edited query text; the box state is untouched."""
        record = {"id": item_id, "query": query, "annotator": annotator, "ts": _now()}
        with self._write_lock():
            self._append(record)
            self._write_snapshots()
        return query

    def get_query(self, item_id: str) -> str | None:
        with self._lock:
            self._refresh_locked()
            return self._queries.get(item_id)

    def delete(self, item_id: str, annotator: str | None = None) -> None:
        record = {"id": item_id, "bbox": None, "annotator": annotator, "ts": _now()}
        with self._write_lock():
            self._append(record)
            self._write_snapshots()

    def _append(self, record: dict) -> None:
        self._append_many([record])

    def _append_many(self, records: list[dict]) -> None:
        # Preserve all original bytes. A damaged/non-newline-terminated tail
        # gets its own line so it cannot swallow the next successful write.
        with self.journal_path.open("a+b") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell():
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")
            for record in records:
                fh.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())

    def _write_snapshots(self) -> None:
        # Caller holds both thread and process locks, so all snapshots reflect
        # the same journal. Readers recover from the journal after any crash.
        self._replay()
        atomic_write_json(self.snapshot_path, self._state)
        atomic_write_json(self.queries_path, self._queries)
        atomic_write_json(self.absent_path, self._absent)
        self._sync_active_dataset()
        self._sync_skeleton()

    def _lookup_skeleton_image(self, item_id: str) -> str:
        if self._skeleton_images is None:
            self._skeleton_images = {}
            if self.skeleton_path and self.skeleton_path.is_file():
                try:
                    skel = json.loads(self.skeleton_path.read_text(encoding="utf-8"))
                    self._skeleton_images = {
                        it["id"]: it["image"]
                        for it in skel.get("items", [])
                        if "id" in it and "image" in it
                    }
                except Exception:
                    pass
        if item_id in self._skeleton_images:
            return self._skeleton_images[item_id]
        parts = item_id.split("#")[0].split("_")
        if len(parts) >= 2:
            return f"Raw/{parts[0]}/color/{parts[1]}.png"
        return ""

    def _sync_active_dataset(self) -> None:
        if not self.active_dataset_path or not self.active_dataset_path.is_file():
            return
        from aicomp_grounding.sharding import group_keys_by_scene
        try:
            active_dataset = json.loads(self.active_dataset_path.read_text(encoding="utf-8"))
        except Exception:
            return
        data = active_dataset.get("data", {})
        changed = False
        for item_id in self._touched:
            if item_id in self._state and item_id in self._queries:
                bbox = self._state[item_id]
                query = self._queries[item_id]
                if bbox and query:
                    if item_id not in data:
                        image = self._lookup_skeleton_image(item_id)
                        if image:
                            data[item_id] = {
                                "visible": image,
                                "infrared": image.replace("/color/", "/infrared/"),
                                "depth": image.replace("/color/", "/depth/"),
                                "bbox": bbox,
                                "width": 1920,
                                "height": 1080,
                                "query": query,
                            }
                            changed = True
                    else:
                        entry = data[item_id]
                        if entry.get("bbox") != bbox or entry.get("query") != query:
                            entry["bbox"] = bbox
                            entry["query"] = query
                            changed = True
            elif item_id in self._absent and item_id in data:
                del data[item_id]
                changed = True

        if changed:
            active_dataset["data"] = data
            active_dataset["metadata"]["sample_count"] = len(data)
            active_dataset["metadata"]["sequence_count"] = len(group_keys_by_scene(list(data), data))
            atomic_write_json(self.active_dataset_path, active_dataset)

    def _sync_skeleton(self) -> None:
        if not self.skeleton_path or not self.skeleton_path.is_file():
            return
        try:
            skel = json.loads(self.skeleton_path.read_text(encoding="utf-8"))
        except Exception:
            return
        items = skel.get("items", [])
        changed = False
        for it in items:
            item_id = it.get("id")
            if not item_id:
                continue
            if item_id in self._state or item_id in self._queries:
                cur_bbox = self._state.get(item_id, it.get("bbox"))
                cur_query = self._queries.get(item_id, it.get("query"))
                cur_status = "annotated" if (cur_bbox is not None and cur_query) else it.get("status", "pending")
                if it.get("bbox") != cur_bbox or it.get("query") != cur_query or it.get("status") != cur_status:
                    it["bbox"] = cur_bbox
                    it["query"] = cur_query
                    it["status"] = cur_status
                    changed = True
        if changed:
            atomic_write_json(self.skeleton_path, skel)

