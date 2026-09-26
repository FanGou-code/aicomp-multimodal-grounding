"""Run-configuration contract: YAML sections, precedence, and rejection rules."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.config import load_run_config, merge_run_config


def _write_yaml(directory: Path, text: str, name: str = "run.yaml") -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class LoadRunConfigTests(unittest.TestCase):
    def test_reads_both_sections(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_yaml(
                Path(td),
                "run:\n  model: qwen3vl\nhyperparameters:\n  epochs: 3\n",
            )
            config = load_run_config(path)
            self.assertEqual(config["run"], {"model": "qwen3vl"})
            self.assertEqual(config["hyperparameters"], {"epochs": 3})

    def test_missing_section_defaults_to_empty_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            config = load_run_config(_write_yaml(Path(td), "run:\n  model: qwen3vl\n"))
            self.assertEqual(config["run"], {"model": "qwen3vl"})
            self.assertEqual(config["hyperparameters"], {})

    def test_unknown_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_yaml(Path(td), "run: {}\ntraining: {}\n")
            with self.assertRaisesRegex(ValueError, "unknown sections"):
                load_run_config(path)

    def test_non_mapping_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_yaml(Path(td), "- item\n")
            with self.assertRaisesRegex(ValueError, "YAML mapping"):
                load_run_config(path)

    def test_non_mapping_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_yaml(Path(td), "run:\n  - item\n")
            with self.assertRaisesRegex(ValueError, "must be a mapping"):
                load_run_config(path)


class MergeRunConfigTests(unittest.TestCase):
    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--data-dir", type=Path, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        return parser

    def _merge(self, args, config: dict, *, hyperparameter_keys=("batch_size",)) -> None:
        merge_run_config(
            args,
            config,
            parser=self._parser(),
            run_keys=("model", "data_dir"),
            hyperparameter_keys=hyperparameter_keys,
        )

    def test_yaml_fills_unspecified_and_cli_wins(self):
        args = self._parser().parse_args(["--model", "glm46v"])
        self._merge(
            args,
            {
                "run": {"model": "qwen3vl", "data_dir": "data"},
                "hyperparameters": {"batch_size": 2},
            },
        )
        self.assertEqual(args.model, "glm46v")
        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.batch_size, 2)

    def test_unknown_keys_are_rejected(self):
        args = self._parser().parse_args([])
        with self.assertRaisesRegex(ValueError, "unknown run keys"):
            self._merge(args, {"run": {"model_name": "x"}, "hyperparameters": {}})
        with self.assertRaisesRegex(ValueError, "unknown hyperparameters keys"):
            self._merge(args, {"run": {}, "hyperparameters": {"lr": 1e-4}})

    def test_empty_allowed_hyperparameter_set_rejects_every_key(self):
        args = self._parser().parse_args([])
        with self.assertRaisesRegex(ValueError, "unknown hyperparameters keys"):
            self._merge(
                args,
                {"run": {}, "hyperparameters": {"batch_size": 2}},
                hyperparameter_keys=(),
            )


class EntrypointConfigTests(unittest.TestCase):
    def test_train_entrypoint_reads_yaml_and_applies_fallbacks(self):
        from tools.train import parse_args

        with tempfile.TemporaryDirectory() as td:
            path = _write_yaml(
                Path(td),
                "run:\n  annotation_run_id: annot_x\n  model: qwen3_5\n"
                "hyperparameters:\n  epochs: 2\n",
            )
            args = parse_args(["--config", str(path)])
            self.assertEqual(args.annotation_run_id, "annot_x")
            self.assertEqual(args.model, "qwen3_5")
            self.assertEqual(args.epochs, 2)
            self.assertEqual(args.data_dir, Path("data"))
            self.assertEqual(args.num_workers, 0)
            self.assertTrue(args.resume)

    def test_train_cli_overrides_yaml_and_run_id_is_required(self):
        from tools.train import parse_args

        with tempfile.TemporaryDirectory() as td:
            path = _write_yaml(
                Path(td),
                "run:\n  annotation_run_id: annot_x\n  model: qwen3_5\n",
            )
            args = parse_args(["--config", str(path), "--model", "glm46v"])
            self.assertEqual(args.model, "glm46v")
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_train_yaml_config_allows_omitting_annotation_run_id_for_active_dataset(self):
        from tools.train import parse_args

        with tempfile.TemporaryDirectory() as td:
            path = _write_yaml(
                Path(td),
                "run:\n  model: qwen3_5\n",
            )
            args = parse_args(["--config", str(path)])
            self.assertIsNone(args.annotation_run_id)
            self.assertEqual(args.model, "qwen3_5")

            empty_str_path = _write_yaml(
                Path(td),
                "run:\n  annotation_run_id: ''\n  model: qwen3_5\n",
                name="empty.yaml",
            )
            args_empty = parse_args(["--config", str(empty_str_path)])
            self.assertIsNone(args_empty.annotation_run_id)

        # Also verify factory configs parse cleanly out of the box
        for config_name in ("train_qwen3vl.yaml", "train_qwen3_5.yaml", "train_glm46v.yaml"):
            cfg_path = Path("configs") / config_name
            args = parse_args(["--config", str(cfg_path)])
            self.assertIsNone(args.annotation_run_id)


    def test_infer_entrypoint_reads_yaml_and_rejects_hyperparameters(self):
        from tools.infer import parse_args

        with tempfile.TemporaryDirectory() as td:
            path = _write_yaml(Path(td), "run:\n  batch_size: 2\n")
            args = parse_args(["--config", str(path)])
            self.assertEqual(args.batch_size, 2)
            self.assertEqual(args.max_pixels, 3072 * 28 * 28)
            self.assertEqual(args.batch_save, 50)

            bad = _write_yaml(Path(td), "hyperparameters:\n  epochs: 1\n", name="bad.yaml")
            with self.assertRaisesRegex(ValueError, "unknown hyperparameters keys"):
                parse_args(["--config", str(bad)])


if __name__ == "__main__":
    unittest.main()
