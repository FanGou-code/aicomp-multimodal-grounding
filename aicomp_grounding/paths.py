"""Repository-relative path policy for the entrypoints.

The repository root is the portable unit copied to a workstation or GPU
environment.  Dataset files, generated annotation artifacts, and run outputs
live in separate roots below it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Artifact families under the output root. This is the single source of the
#: layout documented in ``docs/data-contract.md``; entrypoints take their
#: defaults from here instead of spelling the directory out again.
OUTPUT_FAMILIES = frozenset({
    "annotations",
    "training",
    "inference",
    "fusion",
    "submission",
})


def output_dir(family: str, *, root: str | Path = "outputs") -> Path:
    """``<root>/<family>`` for one artifact family; unknown families are refused."""
    if family not in OUTPUT_FAMILIES:
        raise ValueError(f"Unknown output family: {family!r}")
    return Path(root) / family


def resolve_from_root(path: str | Path, root: str | Path) -> Path:
    """Resolve a user path relative to an explicit project root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(root).resolve() / candidate).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical repository paths used by the entrypoints."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path = ".") -> "ProjectPaths":
        return cls(root=Path(root).resolve())
