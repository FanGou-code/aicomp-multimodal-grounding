"""Contract test: the Modal shell must supply every option run_cli() reads.

``cloud/infer.py`` builds an ``argparse.Namespace`` from ``INFERENCE_DEFAULTS``
plus CLI flags, then hands it to ``offline.infer.run_cli``. A missing default
only surfaces at runtime inside a Modal container (no GPU locally), so this
test compares the attribute names accessed by ``run_cli`` against the cloud
defaults statically via ``ast``.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Provided by the shell itself rather than through INFERENCE_DEFAULTS.
SHELL_PROVIDED = {"project_root", "annotation_run_id"}


def _run_cli_attributes() -> set[str]:
    tree = ast.parse((ROOT / "offline" / "infer.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_cli":
            return {
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "args"
            }
    raise AssertionError("run_cli() not found in offline/infer.py")


def _cloud_defaults() -> set[str]:
    tree = ast.parse((ROOT / "cloud" / "infer.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "INFERENCE_DEFAULTS" and isinstance(node.value, ast.Dict):
                return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError("INFERENCE_DEFAULTS not found in cloud/infer.py")


class RunCliContractTests(unittest.TestCase):
    def test_cloud_defaults_cover_every_option_run_cli_reads(self):
        missing = _run_cli_attributes() - _cloud_defaults() - SHELL_PROVIDED
        self.assertEqual(set(), missing, f"cloud/infer.py must default {sorted(missing)}")

    def test_max_pixels_default_matches_adapter_default(self):
        """The cloud default must equal the qwen3vl adapter default (no id drift)."""
        source = (ROOT / "cloud" / "infer.py").read_text(encoding="utf-8")
        self.assertIn("3072 * 28 * 28", source)
        adapter = (ROOT / "aicomp_grounding" / "models" / "qwen3vl.py").read_text(encoding="utf-8")
        self.assertIn("MAX_PIXELS = 3072 * 28 * 28", adapter)


if __name__ == "__main__":
    unittest.main()
