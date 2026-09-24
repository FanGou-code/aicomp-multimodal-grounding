"""Load the ordinal prompt files.

The prompt text lives in ``prompts/*.md`` rather than in code, so prompt edits
are text edits.  A missing or malformed file is a hard error: in the submission
chain a silent fallback would quietly change behaviour, and a missing file is a
packaging bug that should surface immediately.

Each file carries both parts, delimited by ``[system]`` and ``[user]``, so the
hash of the file covers everything that can change the model's behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

PARSE = "parse"
ENUMERATE = "enumerate"
PROMPT_NAMES = (PARSE, ENUMERATE)

_SYSTEM_MARK = "[system]"
_USER_MARK = "[user]"


def read_prompt_file(name: str) -> str:
    """Return the raw prompt file text, raising when it is missing or empty."""
    if name not in PROMPT_NAMES:
        raise ValueError(f"Unknown ordinal prompt {name!r}; expected one of {PROMPT_NAMES}")
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Ordinal prompt file is missing: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Ordinal prompt file is empty: {path}")
    return text


def load_prompt(name: str) -> tuple[str, str]:
    """Split one prompt file into its ``(system, user)`` parts."""
    text = read_prompt_file(name)
    if not text.startswith(_SYSTEM_MARK):
        raise ValueError(f"Ordinal prompt {name!r} must start with {_SYSTEM_MARK}")
    head, separator, tail = text.partition(_USER_MARK)
    if not separator:
        raise ValueError(f"Ordinal prompt {name!r} is missing the {_USER_MARK} marker")
    system = head[len(_SYSTEM_MARK):].strip()
    user = tail.strip()
    if not system or not user:
        raise ValueError(f"Ordinal prompt {name!r} has an empty system or user part")
    return system, user


def prompt_hashes() -> dict[str, str]:
    """One sha256 per prompt file, over the whole file including both parts."""
    return {
        name: hashlib.sha256(read_prompt_file(name).encode("utf-8")).hexdigest()
        for name in PROMPT_NAMES
    }


def prompts_fingerprint() -> str:
    """Single fingerprint covering every ordinal prompt, for the run identity."""
    encoded = json.dumps(prompt_hashes(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
