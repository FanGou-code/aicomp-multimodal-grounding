"""Contract test: the Modal shell must supply every option run_cli() reads.

``cloud/{infer,train}.py`` builds an ``argparse.Namespace`` from a defaults
dict plus CLI flags, then hands it to ``offline.{infer,train}.run_cli``. A
missing default only surfaces at runtime inside a Modal container (no GPU
locally), so these tests compare the attribute names accessed by ``run_cli``
against the cloud defaults statically via ``ast``.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Provided by the shell itself rather than through its *_DEFAULTS dict.
SHELL_PROVIDED = {
    "infer": {"project_root", "annotation_run_id"},
    "train": {"project_root", "annotation_run_id"},
}


def _run_cli_attributes(cli_file: str, fn_name: str) -> set[str]:
    tree = ast.parse((ROOT / "offline" / cli_file).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return {
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "args"
            }
    raise AssertionError(f"{fn_name}() not found in offline/{cli_file}")


def _cloud_defaults(cloud_file: str, const_name: str) -> set[str]:
    tree = ast.parse((ROOT / "cloud" / cloud_file).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == const_name and isinstance(node.value, ast.Dict):
                return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError(f"{const_name} not found in cloud/{cloud_file}")


class RunCliContractTests(unittest.TestCase):
    def test_infer_defaults_cover_every_option_run_cli_reads(self):
        missing = (
            _run_cli_attributes("infer.py", "run_cli")
            - _cloud_defaults("infer.py", "INFERENCE_DEFAULTS")
            - SHELL_PROVIDED["infer"]
        )
        self.assertEqual(set(), missing, f"cloud/infer.py must default {sorted(missing)}")

    def test_train_defaults_cover_every_option_run_cli_reads(self):
        missing = (
            _run_cli_attributes("train.py", "run_cli")
            - _cloud_defaults("train.py", "TRAINING_DEFAULTS")
            - SHELL_PROVIDED["train"]
        )
        self.assertEqual(set(), missing, f"cloud/train.py must default {sorted(missing)}")

    def test_max_pixels_default_matches_adapter_default(self):
        """The cloud default must equal the qwen3vl adapter default (no id drift)."""
        source = (ROOT / "cloud" / "infer.py").read_text(encoding="utf-8")
        self.assertIn("3072 * 28 * 28", source)
        adapter = (ROOT / "aicomp_grounding" / "models" / "qwen3vl.py").read_text(encoding="utf-8")
        self.assertIn("MAX_PIXELS = 3072 * 28 * 28", adapter)


if __name__ == "__main__":
    unittest.main()
