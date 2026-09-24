"""Tests for the portable repository path contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.paths import (
    OUTPUT_FAMILIES,
    output_dir,
    resolve_from_root,
)


class ProjectPathsTests(unittest.TestCase):
    def test_output_families_are_repo_relative_and_validated(self):
        for family in OUTPUT_FAMILIES:
            self.assertEqual(output_dir(family), Path("outputs") / family)
        with self.assertRaises(ValueError):
            output_dir("nope")

    def test_relative_arguments_resolve_from_project_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            self.assertEqual(resolve_from_root("data", root), root / "data")
            absolute = root / "other"
            self.assertEqual(resolve_from_root(absolute, root), absolute)


if __name__ == "__main__":
    unittest.main()
