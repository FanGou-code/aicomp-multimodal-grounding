"""Runtime identity contract for GPU training metadata."""

from __future__ import annotations

import platform
import unittest

from aicomp_grounding.config import RUNTIME_PYTHON_VERSION, current_runtime_packages


class ConfigRuntimeTests(unittest.TestCase):
    def test_python_version_is_current(self):
        self.assertEqual(RUNTIME_PYTHON_VERSION, platform.python_version())

    def test_runtime_packages_keep_required_baseline(self):
        packages = current_runtime_packages()
        self.assertTrue(any(package.startswith("torch==") for package in packages))
        self.assertTrue(
            any(package.startswith("transformers==") for package in packages)
        )
