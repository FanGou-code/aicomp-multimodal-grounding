"""Tests for the portable repository path contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.paths import ProjectPaths, resolve_from_root


class ProjectPathsTests(unittest.TestCase):
    def test_paths_are_repository_relative(self):
        with tempfile.TemporaryDirectory() as td:
            paths = ProjectPaths.from_root(td)
            self.assertEqual(paths.data, Path(td).resolve() / "data")
            self.assertEqual(
                paths.annotation_artifact("annot_x", "val"),
                Path(td).resolve()
                / "outputs/annotations/annot_x/val/approved.json",
            )
            self.assertEqual(
                paths.submission_template,
                Path(td).resolve() / "data/Test/queries/queries.json",
            )

    def test_relative_arguments_resolve_from_project_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            self.assertEqual(resolve_from_root("data", root), root / "data")
            absolute = root / "other"
            self.assertEqual(resolve_from_root(absolute, root), absolute)

    def test_invalid_split_is_rejected(self):
        paths = ProjectPaths.from_root(".")
        with self.assertRaises(ValueError):
            paths.dataset_index("test_template")
        with self.assertRaises(ValueError):
            paths.annotation_artifact("annot_x", "test")


if __name__ == "__main__":
    unittest.main()
