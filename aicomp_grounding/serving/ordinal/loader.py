"""Load the ordinal prompt files.

The prompt text lives in ``prompts/*.md`` rather than in code, so prompt edits
are text edits.  Each file carries both parts, delimited by ``[system]`` and
``[user]``, so the hash of the file covers everything that can change the
model's behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aicomp_grounding.serving.prompt_files import read_prompt_text, split_prompt_pair

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

PARSE = "parse"
ENUMERATE = "enumerate"
PROMPT_NAMES = (PARSE, ENUMERATE)


def _prompt_path(name: str) -> Path:
    if name not in PROMPT_NAMES:
        raise ValueError(f"Unknown ordinal prompt {name!r}; expected one of {PROMPT_NAMES}")
    return PROMPT_DIR / f"{name}.md"


def read_prompt_file(name: str) -> str:
    """Return the raw prompt file text, raising when it is missing or empty."""
    return read_prompt_text(_prompt_path(name), label=f"Ordinal prompt {name!r}")


def load_prompt(name: str) -> tuple[str, str]:
    """Split one prompt file into its ``(system, user)`` parts."""
    label = f"Ordinal prompt {name!r}"
    return split_prompt_pair(read_prompt_file(name), label=label)


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
