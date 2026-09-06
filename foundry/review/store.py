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


Vendored from gt-annotator (github.com/FanGou-code/gt-annotator,
MIT License, (c) 2026 FanGou-code) - adapted for query-foundry review."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

JOURNAL_NAME = "annotations.jsonl"
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
        self.absent_path = self.data_dir / ABSENT_SNAPSHOT_NAME
        self._lock = threading.Lock()
        self._state: dict[str, list[float]] = {}
        self._meta: dict[str, dict] = {}
        self._absent: dict[str, str] = {}
        self._journal_sig: tuple[int, int, int] | None = None
        self._replay()

    # -- journal replay ----------------------------------------------------

    def _read_journal_state(self) -> tuple[dict[str, list[float]], dict[str, dict], dict[str, str]]:
        state: dict[str, list[float]] = {}
        meta: dict[str, dict] = {}
        absent: dict[str, str] = {}
        if not self.journal_path.is_file():
            return state, meta, absent
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
                bbox = record.get("bbox")
                annotator = record.get("annotator")
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
        return state, meta, absent

    def _journal_signature(self) -> tuple[int, int, int] | None:
        try:
            st = self.journal_path.stat()
        except OSError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _replay(self) -> None:
        self._state, self._meta, self._absent = self._read_journal_state()
        self._journal_sig = self._journal_signature()

    def _refresh_locked(self) -> None:
        """Hot-reload in-memory state if another process appended to the journal."""
        for _ in range(5):
            sig = self._journal_signature()
            if sig == self._journal_sig:
                return
            state, meta, absent = self._read_journal_state()
            # Concurrent writers may have appended while we replayed; only
            # commit the replay if the file is unchanged since it started,
            # otherwise the new tail would be swallowed by this signature.
            if self._journal_signature() == sig:
                self._state, self._meta, self._absent = state, meta, absent
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

    def all_boxes(self) -> dict[str, list[float]]:
        with self._lock:
            self._refresh_locked()
            return {item_id: list(bbox) for item_id, bbox in self._state.items()}

    def absent_items(self) -> dict[str, str]:
        with self._lock:
            self._refresh_locked()
            return dict(self._absent)

    def annotated_count(self) -> int:
        with self._lock:
            self._refresh_locked()
            return len(self._state)

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

    def set_absent(self, item_id: str, annotator: str) -> dict:
        """Record an absence verdict, e.g. human confirmation of an AI absence."""
        record = {
            "id": item_id,
            "bbox": None,
            "annotator": f"{annotator}{ABSENT_SUFFIX}",
            "ts": _now(),
        }
        with self._lock:
            self._append(record)
            self._state.pop(item_id, None)
            self._meta[item_id] = {"annotator": record["annotator"], "ts": record["ts"]}
            self._absent[item_id] = record["annotator"]
            self._write_snapshots()
        return {"id": item_id, "bbox": None, "annotator": record["annotator"]}

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
        absent: dict[str, str] = {}
        for _ in range(5):
            sig = self._journal_signature()
            state, _meta, absent = self._read_journal_state()
            if self._journal_signature() == sig:
                break
        _atomic_write_json(self.snapshot_path, state)
        _atomic_write_json(self.absent_path, absent)
