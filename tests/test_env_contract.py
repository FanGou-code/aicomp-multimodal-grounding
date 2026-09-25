"""Environment declaration contract: pyproject.toml is the single source.

These tests pin the invariant that the model-library versions recorded in
``config.RUNTIME_PACKAGES`` match ``pyproject.toml`` exactly, and that torch
and torchvision are declared as exact pins so CUDA wheel variants
(cu130/cu128) are selected by ``tools/setup_colab.sh`` rather than by the
range resolver.
"""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from aicomp_grounding.config import RUNTIME_PACKAGES

ROOT = Path(__file__).resolve().parent.parent

_MODEL_LIBS = ("transformers", "peft", "accelerate", "qwen-vl-utils")


def _load_dependencies() -> list[str]:
    with open(ROOT / "pyproject.toml", "rb") as file:
        return tomllib.load(file)["project"]["dependencies"]


class EnvContractTests(unittest.TestCase):
    def test_model_library_pins_match_pyproject(self):
        deps = _load_dependencies()
        pyproject_pins = {d for d in deps if d.split("==")[0] in _MODEL_LIBS}
        config_pins = {p for p in RUNTIME_PACKAGES if p.split("==")[0] in _MODEL_LIBS}
        self.assertEqual(pyproject_pins, config_pins)

    def test_torch_and_torchvision_are_exact_pins(self):
        deps = _load_dependencies()
        torch_deps = [d for d in deps if d.startswith("torch")]
        self.assertTrue(torch_deps)
        self.assertTrue(all("==" in d for d in torch_deps), torch_deps)

    def test_legacy_requirement_files_removed(self):
        self.assertFalse((ROOT / "envs" / "gpu.txt").exists())
        self.assertFalse((ROOT / "requirements-lock.txt").exists())
        self.assertFalse((ROOT / "requirements.txt").exists())


if __name__ == "__main__":
    unittest.main()
