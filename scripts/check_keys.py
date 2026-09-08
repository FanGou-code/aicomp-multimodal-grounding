"""Probe every API key in the key file and classify its health.

Admin-run tool: it performs REAL API calls (one per key), so per the
repository red line it is executed by the administrator, never by the agent.

Two probe modes:

- ``realistic`` (default): one census-shaped findall call — a 1080p marked
  frame, the panoramic prompt, max_tokens=2048. Consumes roughly the same
  tokens as one real census request, so keys whose remaining balance cannot
  sustain the real workload fail here (402) instead of dying mid-batch.
  The response's billed token usage is reported per key.
- ``minimal``: one 1-token "hi" — near-zero cost liveness check only; a key
  with almost no balance can pass it.

Classification by HTTP status:
  200          OK                     alive
  401          invalid/revoked        dead
  402          balance exhausted      dead until top-up / quota reset
  403          forbidden              dead
  429          rate limited           fake-dead — alive, retest later
  5xx/timeout  provider/network       fake-dead — alive, retest later

Non-OK keys are retested once after a pause to separate transient failures
from persistent ones. Commented-out lines (``#sk-xxx``) are probed too, so a
parked key can be checked for revival. Keys are never printed — only their
1-based line index in the file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from foundry.pipeline.views import build_marked_annotation_view, jpeg_data_url  # noqa: E402
from foundry.pipeline.census import findall_messages  # noqa: E402
from foundry.utils import ANNOTATION_API_BASE_URL, ANNOTATION_MODEL_NAME, ANNOTATION_TEMPERATURE  # noqa: E402

REALISTIC_MAX_TOKENS = 2048

VERDICTS = {
    200: "OK",
    401: "DEAD (invalid/revoked)",
    402: "DEAD (balance exhausted — revives on top-up/reset)",
    403: "DEAD (forbidden)",
    429: "FAKE-DEAD (rate limited — alive, retest later)",
}


def read_key_lines(path: Path) -> list[tuple[int, str, bool]]:
    """Return (line_no, key, was_commented) for every probeable key line.

    Convention: a commented line yields a probeable parked key when its
    comment-stripped body is a single whitespace-free token (``#sk-xxx``);
    prose notes contain spaces and are ignored.
    """
    entries: list[tuple[int, str, bool]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        commented = stripped.startswith("#")
        body = stripped[1:] if commented else stripped
        key = body.split("#", 1)[0].strip()
        if key and not any(ch.isspace() for ch in key):
            entries.append((line_no, key, commented))
    return entries


def classify_status(status: int | None, error_text: str) -> str:
    if status is None:
        return "FAKE-DEAD (network/timeout — alive, retest later)"
    if status in VERDICTS:
        return VERDICTS[status]
    if 500 <= status <= 599:
        return "FAKE-DEAD (server error — alive, retest later)"
    return f"UNEXPECTED HTTP {status}: {error_text[:120]}"


def build_payload(*, probe: str, data_root: Path, model: str) -> dict:
    """Build the request payload shared by every key probe."""
    if probe == "minimal":
        return {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "thinking": {"type": "disabled"},
        }
    index_path = data_root / "indexes" / "train.json"
    if not index_path.is_file():
        raise SystemExit(f"train index not found at {index_path} (realistic probe needs it; try --probe minimal)")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sample_id = sorted(index)[0]
    item = index[sample_id]
    image = Image.open(data_root / item["visible"]).convert("RGB")
    marked = build_marked_annotation_view(image, item["bbox"])
    return {
        "model": model,
        "messages": findall_messages(jpeg_data_url(marked)),
        "max_tokens": REALISTIC_MAX_TOKENS,
        "temperature": ANNOTATION_TEMPERATURE,
        "thinking": {"type": "disabled"},
    }


def probe_key(key: str, *, payload: dict, base_url: str, timeout: float) -> tuple[int | None, str, dict]:
    """One probe request; return (http_status_or_None, detail, usage)."""
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
            return response.status, "", body.get("usage", {})
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            detail = ""
        return exc.code, detail, {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, str(exc)[:200], {}


def describe_usage(usage: dict) -> str:
    if not usage:
        return ""
    return (f" [billed: prompt {usage.get('prompt_tokens', '?')} tok, "
            f"completion {usage.get('completion_tokens', '?')} tok]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(PROJECT_ROOT / "keys" / "api_keys.txt"))
    parser.add_argument("--probe", choices=("realistic", "minimal"), default="realistic")
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="dataset root for the realistic probe (unused by --probe minimal)",
    )
    parser.add_argument("--pause", type=float, default=20.0, help="seconds before retesting non-OK keys")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"key file not found: {path}")
        return 2
    entries = read_key_lines(path)
    if not entries:
        print(f"no keys in {path}")
        return 2
    payload = build_payload(probe=args.probe, data_root=Path(args.data_root), model=ANNOTATION_MODEL_NAME)
    print(f"probing {len(entries)} keys from {path} | probe={args.probe} model={ANNOTATION_MODEL_NAME} (1 call each)")

    results: dict[int, tuple[int | None, str]] = {}
    for line_no, key, commented in entries:
        status, detail, usage = probe_key(key, payload=payload, base_url=ANNOTATION_API_BASE_URL, timeout=args.timeout)
        results[line_no] = (status, detail)
        flag = " (commented out in file)" if commented else ""
        print(f"  line {line_no}{flag}: HTTP {status} -> {classify_status(status, detail)}{describe_usage(usage)}", flush=True)
        time.sleep(0.5)

    retest = [ln for ln, (status, _) in results.items() if status != 200]
    if retest and args.pause > 0:
        print(f"retesting {len(retest)} non-OK key(s) after {args.pause:.0f}s to separate transient from persistent...")
        time.sleep(args.pause)
        for line_no, key, commented in entries:
            if line_no not in retest:
                continue
            status, detail, _ = probe_key(key, payload=payload, base_url=ANNOTATION_API_BASE_URL, timeout=args.timeout)
            first = results[line_no]
            results[line_no] = status if status == 200 else first
            note = f"was {first[0]}, now" if status == 200 else "still"
            print(f"  line {line_no}: {note} HTTP {status} -> {classify_status(results[line_no][0], detail)}", flush=True)

    ok = sum(1 for status, _ in results.values() if status == 200)
    print(f"summary: {ok}/{len(results)} alive")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
