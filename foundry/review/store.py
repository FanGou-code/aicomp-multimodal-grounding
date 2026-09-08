"""Crash-safe annotation store.

Durable state = append-only JSONL journal (one record per PUT/DELETE).
``annotations.predictions.json`` is a consolidated snapshot in the main
project's prediction-file shape (``{item_id: [x1, y1, x2, y2]}``, normalized
0-1 XYXY, annotated items only) and is rewritten atomically on every change.
Absence verdicts are journaled as ``bbox: null`` records whose ``annotator``
ends in ``:absent`` (e.g. ``glm-4.6v:absent`` pending review, ``fang0:absent``
human-confirmed) and consolidated into ``annotations.absent.json`` as
``{item_id: annotator}``. Recovery = replay the journal; a torn trailing line
from a crash is ignored.

Snapshots are always re-derived from a full journal replay at write time, so
concurrent writer processes (auto_annotate.py and server.py) never drop each
other's entries. Readers re-replay automatically whenever the journal file
changes on disk, which hot-reloads annotations written by other processes.
Query edits are journaled as bbox-less records and replay as text + annotator
updates without touching box state.


Adapted from gt-annotator (upstream gt-annotator project,
MIT License, (c) 2026 FanGou-code) - adapted for query-foundry review."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

JOURNAL_NAME = "annotations.jsonl"
QUERY_SNAPSHOT_NAME = "annotations.queries.json"
SNAPSHOT_NAME = "annotations.predictions.json"
ABSENT_SNAPSHOT_NAME = "annotations.absent.json"
ABSENT_SUFFIX = ":absent"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _is_absent_annotator(annotator: object) -> bool:
    return isinstance(annotator, str) and annotator.endswith(ABSENT_SUFFIX)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class AnnotationStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.journal_path = self.data_dir / JOURNAL_NAME
        self.snapshot_path = self.data_dir / SNAPSHOT_NAME
        self.queries_path = self.data_dir / QUERY_SNAPSHOT_NAME
        self.absent_path = self.data_dir / ABSENT_SNAPSHOT_NAME
        self._lock = threading.Lock()
        self._state: dict[str, list[float]] = {}
        self._queries: dict[str, str] = {}
        self._meta: dict[str, dict] = {}
        self._absent: dict[str, str] = {}
        self._journal_sig: tuple[int, int, int] | None = None
        self._replay()

    # -- journal replay ----------------------------------------------------

    def _read_journal_state(self) -> tuple[dict[str, list[float]], dict[str, str], dict[str, dict], dict[str, str]]:
        state: dict[str, list[float]] = {}
        queries: dict[str, str] = {}
        meta: dict[str, dict] = {}
        absent: dict[str, str] = {}
        if not self.journal_path.is_file():
            return state, queries, meta, absent
        with self.journal_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail from a crash; journal remains truth
                item_id = record.get("id")
                if not isinstance(item_id, str) or not item_id:
                    continue
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
        return state, queries, meta, absent

    def _journal_signature(self) -> tuple[int, int, int] | None:
        try:
            st = self.journal_path.stat()
        except OSError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _replay(self) -> None:
        self._state, self._queries, self._meta, self._absent = self._read_journal_state()
        self._journal_sig = self._journal_signature()

    def _refresh_locked(self) -> None:
        """Hot-reload in-memory state if another process appended to the journal."""
        for _ in range(5):
            sig = self._journal_signature()
            if sig == self._journal_sig:
                return
            state, queries, meta, absent = self._read_journal_state()
            # Concurrent writers may have appended while we replayed; only
            # commit the replay if the file is unchanged since it started,
            # otherwise the new tail would be swallowed by this signature.
            if self._journal_signature() == sig:
                self._state, self._queries, self._meta, self._absent = state, queries, meta, absent
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

    def set(self, item_id: str, bbox: list[float], annotator: str | None = None) -> list[float]:
        bbox = [float(v) for v in bbox]
        record = {"id": item_id, "bbox": bbox, "annotator": annotator, "ts": _now()}
        with self._lock:
            self._append(record)
            self._state[item_id] = bbox
            self._meta[item_id] = {"annotator": annotator, "ts": record["ts"]}
            self._absent.pop(item_id, None)
            # Deliberately do NOT advance _journal_sig here: other processes
            # may have appended before our append, and stat-ing now would
            # mark those unread records as seen. The next read re-replays.
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
        with self._lock:
            self._refresh_locked()
            if not seeds:
                return 0
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as fh:
                for item_id, bbox, annotator in seeds:
                    record = {
                        "id": item_id,
                        "bbox": [float(v) for v in bbox],
                        "annotator": annotator,
                        "ts": _now(),
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self._state[item_id] = record["bbox"]
                    self._meta[item_id] = {"annotator": annotator, "ts": record["ts"]}
                    self._absent.pop(item_id, None)
                fh.flush()
                os.fsync(fh.fileno())
            # Like set(), do not advance _journal_sig: the next read re-replays
            # once and reconciles any interleaved foreign appends.
            self._write_snapshots()
            return len(seeds)

    def set_query(self, item_id: str, query: str, annotator: str | None = None) -> str:
        """Record a human-edited query text; the box state is untouched."""
        record = {"id": item_id, "query": query, "annotator": annotator, "ts": _now()}
        with self._lock:
            self._append(record)
            self._queries[item_id] = query
            self._meta[item_id] = {"annotator": annotator, "ts": record["ts"]}
            self._write_snapshots()
        return query

    def get_query(self, item_id: str) -> str | None:
        with self._lock:
            self._refresh_locked()
            return self._queries.get(item_id)

    def delete(self, item_id: str, annotator: str | None = None) -> None:
        record = {"id": item_id, "bbox": None, "annotator": annotator, "ts": _now()}
        with self._lock:
            self._append(record)
            self._state.pop(item_id, None)
            self._meta.pop(item_id, None)
            self._absent.pop(item_id, None)
            self._write_snapshots()

    def _append(self, record: dict) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _write_snapshots(self) -> None:
        # Caller holds the lock. Re-derive from the journal instead of the
        # in-memory state so entries written by other processes survive.
        state: dict[str, list[float]] = {}
        queries: dict[str, str] = {}
        meta: dict[str, dict] = {}
        absent: dict[str, str] = {}
        for _ in range(5):
            sig = self._journal_signature()
            state, queries, meta, absent = self._read_journal_state()
            if self._journal_signature() == sig:
                break
        # Every snapshot AND the in-memory views must agree with the journal.
        # Writing a stale local queries dict here would let a box-only writer
        # (set/seed/delete) flush another process's query edits out of
        # annotations.queries.json even though the journal still holds them.
        self._state, self._queries, self._meta, self._absent = state, queries, meta, absent
        _atomic_write_json(self.snapshot_path, self._state)
        _atomic_write_json(self.queries_path, self._queries)
        _atomic_write_json(self.absent_path, self._absent)
