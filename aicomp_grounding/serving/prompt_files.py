"""Shared ``[system]`` / ``[user]`` prompt-file parsing.

Prompt texts live in ``prompts/*.md`` files so prompt edits are text edits and
the run identity can hash everything that changes the model's behaviour.  A
missing, empty, or malformed file is a hard error: a silent fallback would
change behaviour without changing the run identity.
"""

from __future__ import annotations

from pathlib import Path

SYSTEM_MARK = "[system]"
USER_MARK = "[user]"


def read_prompt_text(path: Path, *, label: str) -> str:
    """Return the stripped file text, raising when the file is missing or empty."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} prompt file is missing: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} prompt file is empty: {path}")
    return text


def split_prompt_pair(text: str, *, label: str) -> tuple[str, str]:
    """Split one prompt file into its ``(system, user)`` parts."""
    if not text.startswith(SYSTEM_MARK):
        raise ValueError(f"{label} prompt must start with {SYSTEM_MARK}")
    head, separator, tail = text.partition(USER_MARK)
    if not separator:
        raise ValueError(f"{label} prompt is missing the {USER_MARK} marker")
    system = head[len(SYSTEM_MARK):].strip()
    user = tail.strip()
    if not system or not user:
        raise ValueError(f"{label} prompt has an empty system or user part")
    return system, user
