import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.grounding.engine.training_core import (
    _EPOCH_CHECKPOINT_RETENTION,
    _STEP_CHECKPOINT_RETENTION,
    _prune_checkpoints,
)


def _make_checkpoints(root: Path, names: list[str]) -> None:
    for name in names:
        (root / "checkpoints" / name).mkdir(parents=True, exist_ok=True)


def _remaining(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "checkpoints").iterdir())


class PruneCheckpointsTests(unittest.TestCase):
    def test_prune_keeps_newest_step_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_checkpoints(root, [f"step_{i:04d}" for i in (50, 100, 150, 200)])
            _prune_checkpoints(root / "checkpoints", "step_", 2)
            self.assertEqual(_remaining(root), ["step_0150", "step_0200"])

    def test_prune_zero_removes_all_of_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_checkpoints(root, ["step_0050", "step_0100", "epoch_01"])
            _prune_checkpoints(root / "checkpoints", "step_", 0)
            self.assertEqual(_remaining(root), ["epoch_01"])

    def test_prune_epoch_keeps_only_newest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_checkpoints(root, ["epoch_01", "epoch_02", "epoch_03"])
            _prune_checkpoints(root / "checkpoints", "epoch_", 1)
            self.assertEqual(_remaining(root), ["epoch_03"])

    def test_prune_ignores_unrelated_entries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_checkpoints(root, ["step_0050", "epoch_01"])
            (root / "checkpoints" / "stray.txt").write_text("x")
            _prune_checkpoints(root / "checkpoints", "step_", 2)
            self.assertEqual(_remaining(root), ["epoch_01", "step_0050", "stray.txt"])

    def test_prune_missing_directory_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            _prune_checkpoints(Path(td) / "checkpoints", "step_", 2)

    def test_prune_negative_keep_raises(self):
        with self.assertRaises(ValueError):
            _prune_checkpoints(Path("."), "step_", -1)

    def test_retention_constants_allow_resume(self):
        # Resume needs at least one usable step checkpoint plus one spare.
        self.assertGreaterEqual(_STEP_CHECKPOINT_RETENTION, 2)
        # At least one epoch checkpoint must survive for boundary resumes.
        self.assertGreaterEqual(_EPOCH_CHECKPOINT_RETENTION, 1)


if __name__ == "__main__":
    unittest.main()
