"""Probe the injected API key once and classify its health.

Admin-run tool: it performs a REAL API call, so per the repository red line it
is executed by the administrator, never by the agent. The key is read from
``--api-key`` or ``$ANNOTATION_API_KEY`` and is never printed.

The default probe is census-shaped — a 1080p marked frame, the findall prompt,
``max_tokens=2048`` — so a key whose remaining balance cannot sustain the real
workload fails here (402) instead of dying mid-batch. ``--probe minimal`` sends
a 1-token "hi" as a near-zero-cost liveness check.

Classification by HTTP status:
  200          OK                     alive
  401          invalid/revoked        dead
  402          balance exhausted      dead until top-up / quota reset
  403          forbidden              dead
  429          rate limited           fake-dead — alive, retest later
  5xx/timeout  provider/network       fake-dead — alive, retest later
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from aicomp_grounding.annotation.api import resolve_api_key  # noqa: E402
from aicomp_grounding.annotation.census import findall_messages  # noqa: E402
from aicomp_grounding.annotation.imaging import build_marked_annotation_view, jpeg_data_url  # noqa: E402
from aicomp_grounding.annotation.config import ANNOTATION_API_BASE_URL, ANNOTATION_MODEL_NAME, ANNOTATION_TEMPERATURE  # noqa: E402

REALISTIC_MAX_TOKENS = 2048

VERDICTS = {
    200: "OK",
    401: "DEAD (invalid/revoked)",
    402: "DEAD (balance exhausted — revives on top-up/reset)",
    403: "DEAD (forbidden)",
    429: "FAKE-DEAD (rate limited — alive, retest later)",
}


def classify_status(status: int | None, error_text: str) -> str:
    if status is None:
        return f"FAKE-DEAD (transport — {error_text[:60]})" if error_text else "FAKE-DEAD (transport)"
    if status in VERDICTS:
        return VERDICTS[status]
    if 500 <= status < 600:
        return "FAKE-DEAD (provider 5xx — alive, retest later)"
    return f"UNEXPECTED HTTP {status}"


def build_payload(*, probe: str, data_root: Path, model: str, index_dir: Path | None = None) -> dict:
    """The request payload: census-shaped by default, 1-token when minimal."""
    if probe == "minimal":
        return {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "thinking": {"type": "disabled"},
        }
    from aicomp_grounding.annotation.config import resolve_index_dir

    index_path = resolve_index_dir(data_root, index_dir) / "train.json"
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
    parser.add_argument("--api-key", default=None,
                        help="key to probe; defaults to $ANNOTATION_API_KEY")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--probe", choices=("realistic", "minimal"), default="realistic")
    parser.add_argument("--data-root", type=str, default="",
                        help="dataset root for the realistic probe (unused by --probe minimal)")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    key = resolve_api_key(args.api_key)
    payload = build_payload(
        probe=args.probe, data_root=Path(args.data_root), model=ANNOTATION_MODEL_NAME,
        index_dir=args.index_dir,
    )
    print(f"probing the injected key | probe={args.probe} model={ANNOTATION_MODEL_NAME} (1 call)")
    status, detail, usage = probe_key(key, payload=payload, base_url=ANNOTATION_API_BASE_URL, timeout=args.timeout)
    print(f"  HTTP {status} -> {classify_status(status, detail)}{describe_usage(usage)}")
    if detail:
        print(f"  detail: {detail}")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
