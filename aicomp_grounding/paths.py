"""Repository-relative path policy for the entrypoints.

The repository root is the portable unit copied to a workstation or GPU
environment.  Dataset files, generated annotation artifacts, and run outputs
live in separate roots below it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def annotations(self) -> Path:
        return self.outputs / "annotations"

    @property
    def inference(self) -> Path:
        return self.outputs / "inference"

    @property
    def fusion(self) -> Path:
        return self.outputs / "fusion"

    @property
    def submission_template(self) -> Path:
        return self.data / "Test" / "queries" / "queries.json"

    def dataset_index(self, split: str) -> Path:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset split: {split!r}")
        return self.data / "indexes" / f"{split}.json"

    def annotation_artifact(self, annotation_run_id: str, split: str) -> Path:
        if split not in {"train", "val"}:
            raise ValueError(f"Approved annotations do not support split: {split!r}")
        return self.annotations / annotation_run_id / split / "approved.json"
