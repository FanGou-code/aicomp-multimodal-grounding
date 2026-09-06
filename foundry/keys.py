"""API key pool for the annotation pipeline.

Persistent key document (one key per line, '#' comments allowed). Keys come
from different accounts, and the provider limits concurrency per account, so
dispatch rotates across alive keys (one request each, in cursor order) —
this spreads in-flight requests over every account instead of piling them
onto the first key. Retirement semantics are per failure class:

- HTTP 401/402/403 (auth/billing): retired immediately — the key is dead.
  With a ``persist_retire`` callback the key line is commented out in the
  key document (dated reason) so future runs skip it too.
- HTTP 429 (rate limit): NEVER retired for being hot. Strikes accumulate
  cumulatively; only sustained throttling (default 30 strikes) retires a
  key, because a hot key revives the moment the burst passes.
- Transport failures (timeout / URLError): two consecutive failures suspend
  the key for this run — a hanging endpoint is indistinguishable from a
  dead one, and every minute spent on it stalls a worker. Suspended keys
  are NOT written back to the file; re-probe them out-of-band with
  ``scripts/check_keys.py``.

When every key is retired the pool raises :class:`APIKeyPoolExhausted` so
the run fails fast instead of looping — the shard checkpoint is already on
disk, so rerunning the same command resumes from it.

Keys are never logged; only their 1-based index is printed.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

from foundry.api_client import APIError

ENV_KEY_FILE = "ANNOTATION_API_KEY_FILE"
ENV_KEYS = "ANNOTATION_API_KEYS"
DEFAULT_KEY_FILE = Path(__file__).resolve().parents[1] / "keys" / "api_keys.txt"

RATE_LIMIT_STRIKES_BEFORE_RETIRE = 30
TRANSPORT_FAILURES_BEFORE_SUSPEND = 2


class APIKeyPoolExhausted(APIError):
    """Raised when every key in the pool is retired."""


def load_api_keys(path: str | Path | None = None) -> list[str]:
    """Load keys in fixed order.

    Resolution: explicit ``path`` > ``$ANNOTATION_API_KEY_FILE`` >
    ``keys/api_keys.txt`` in the repository > ``$ANNOTATION_API_KEYS``
    (comma-separated). Returns an empty list when no source exists.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    elif os.environ.get(ENV_KEY_FILE, "").strip():
        candidates.append(Path(os.environ[ENV_KEY_FILE].strip()))
    else:
        candidates.append(DEFAULT_KEY_FILE)
    for candidate in candidates:
        if candidate.is_file():
            keys: list[str] = []
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.split("#", 1)[0].strip()
                if line:
                    keys.append(line)
            return keys
    env_value = os.environ.get(ENV_KEYS, "").strip()
    if env_value:
        return [part.strip() for part in env_value.split(",") if part.strip()]
    return []


def comment_out_key(path: Path, index: int, reason: str) -> bool:
    """Comment out the ``index``-th loadable key line with a dated reason.

    Mirrors :func:`load_api_keys` line semantics: a line is loadable when its
    pre-``#`` content is non-empty. Writes atomically and preserves the
    file's 0600 permission. Returns ``True`` when a line was rewritten.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    seen = -1
    target: int | None = None
    for line_no, raw in enumerate(lines):
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        seen += 1
        if seen == index:
            target = line_no
            break
    if target is None:
        return False
    stamp = time.strftime("%Y-%m-%d")
    lines[target] = f"# {lines[target].strip()}  # {stamp} auto-retired: {reason}"
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".api_keys_", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(lines) + "\n")
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return True


class APIKeyPool:
    """Thread-shared rotating key dispatch over per-account key quotas."""

    def __init__(
        self,
        keys: list[str],
        *,
        notify=None,
        persist_retire: Callable[[int, str], None] | None = None,
    ) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if not isinstance(key, str) or not key.strip():
                continue
            key = key.strip()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        if not cleaned:
            raise ValueError("API key pool is empty")
        self._keys = cleaned
        self._retired: set[int] = set()
        self._cursor = 0
        self._transport_failures: dict[int, int] = {}
        self._rate_limit_strikes: dict[int, int] = {}
        self._suspended: set[int] = set()
        self._revived_once: set[int] = set()
        self._lock = threading.Lock()
        self._notify = notify or (lambda message: None)
        self._persist_retire = persist_retire

    @property
    def size(self) -> int:
        return len(self._keys)

    def alive(self) -> int:
        with self._lock:
            return len(self._keys) - len(self._retired)

    def current(self) -> tuple[int, str]:
        """Return ``(index, key)`` of the next alive key, rotating per call.

        Every call advances the cursor, so concurrent workers spread across
        the alive keys instead of concentrating on the first one.
        """
        with self._lock:
            total = len(self._keys)
            for offset in range(total):
                index = (self._cursor + offset) % total
                if index not in self._retired:
                    self._cursor = (index + 1) % total
                    return index, self._keys[index]
            # Pool dry. Hard retirements (auth/billing, sustained 429) stay
            # dead, but transport-suspended keys get exactly one second wind:
            # a provider-wide blip suspends every in-flight key at once, and
            # that should not permanently burn the pool mid-run.
            revivable = [i for i in sorted(self._suspended) if i not in self._revived_once]
            if revivable:
                for index in revivable:
                    self._retired.discard(index)
                    self._suspended.discard(index)
                    self._transport_failures.pop(index, None)
                    self._revived_once.add(index)
                self._cursor = revivable[0]
                self._notify(
                    f"[key-pool] pool exhausted - {len(revivable)} suspended key(s) "
                    "revived for a second pass"
                )
                index = revivable[0]
                self._cursor = (index + 1) % total
                return index, self._keys[index]
        raise APIKeyPoolExhausted(
            "API key pool exhausted: every key is retired. "
            "Add keys to the key file (or $ANNOTATION_API_KEYS) and rerun "
            "the same command to resume from the checkpoint."
        )

    def retire(self, index: int, reason: str) -> None:
        """Retire a key by index (idempotent) and report the switch."""
        with self._lock:
            if index in self._retired or not 0 <= index < len(self._keys):
                return
            self._retired.add(index)
            remaining = len(self._keys) - len(self._retired)
            if remaining:
                self._notify(
                    f"[key-pool] key#{index + 1} retired ({reason}) -> "
                    f"switching to key#{index + 2} ({remaining}/{len(self._keys)} alive)"
                )
            else:
                self._notify(
                    f"[key-pool] key#{index + 1} retired ({reason}) -> "
                    f"pool exhausted ({len(self._keys)} keys)"
                )
            if self._persist_retire is not None:
                try:
                    self._persist_retire(index, reason)
                except OSError as exc:
                    self._notify(f"[key-pool] persisting key#{index + 1} retirement failed: {exc}")

    def note_rate_limited(self, index: int) -> bool:
        """Record one HTTP 429 strike on a key.

        Strikes accumulate for the lifetime of the run; a key that keeps
        answering 429 through every backoff is retired after
        ``RATE_LIMIT_STRIKES_BEFORE_RETIRE`` strikes. Returns ``True`` when
        this call retired the key.
        """
        with self._lock:
            if index in self._retired or not 0 <= index < len(self._keys):
                return False
            strikes = self._rate_limit_strikes.get(index, 0) + 1
            self._rate_limit_strikes[index] = strikes
            if strikes < RATE_LIMIT_STRIKES_BEFORE_RETIRE:
                return False
            self._retired.add(index)
            self._notify(
                f"[key-pool] key#{index + 1} retired (sustained HTTP 429 x{strikes})"
            )
            if self._persist_retire is not None:
                try:
                    self._persist_retire(index, f"sustained HTTP 429 x{strikes}")
                except OSError as exc:
                    self._notify(f"[key-pool] persisting key#{index + 1} retirement failed: {exc}")
            return True

    def note_transport_failure(self, index: int) -> bool:
        """Record one transport failure (timeout / URLError) on a key.

        Consecutive failures suspend the key for the rest of the run after
        ``TRANSPORT_FAILURES_BEFORE_SUSPEND`` — a hanging endpoint stalls a
        worker for the full request timeout, and any healthy response clears
        the counter. Returns ``True`` when this call suspended the key.
        """
        with self._lock:
            if index in self._retired or not 0 <= index < len(self._keys):
                return False
            failures = self._transport_failures.get(index, 0) + 1
            self._transport_failures[index] = failures
            if failures < TRANSPORT_FAILURES_BEFORE_SUSPEND:
                return False
            self._retired.add(index)
            self._suspended.add(index)
            self._notify(
                f"[key-pool] key#{index + 1} suspended after {failures} consecutive "
                "transport failures (re-probe with scripts/check_keys.py)"
            )
            return True

    def note_success(self, index: int) -> None:
        """Clear the consecutive transport-failure counter after a success."""
        with self._lock:
            self._transport_failures.pop(index, None)

    def describe(self) -> str:
        return f"{self.alive()}/{self.size} alive"
