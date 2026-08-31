"""Adapter contract tests + torch-free end-to-end pipeline test (mock model)."""

from __future__ import annotations

import builtins
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from aicomp_grounding.inference_core import evaluate_predictions, load_inference_items
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.models import ADAPTERS, available_models, get_adapter
from aicomp_grounding.models.base import ModelInput, Prediction
from aicomp_grounding.models.groundingdino import (
    normalize_grounding_query,
    select_top_detection,
)
from aicomp_grounding.models.internvl35 import (
    INTERNVL_IMAGE_TOKEN,
    build_grounding_question,
    parse_internvl_box,
)
from aicomp_grounding.models.mock import _stable_box
from aicomp_grounding.models.qwen3vl import (
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_NAME,
    MODEL_REVISION,
)
from aicomp_grounding.prompts import GROUNDING_SYSTEM_PROMPT, grounding_prompt_hash
from aicomp_grounding.submission import build_submission


class RegistryTests(unittest.TestCase):
    def test_registry_exposes_expected_models(self):
        self.assertEqual(
            available_models(),
            ["groundingdino", "internvl35", "mock", "qwen3_8", "qwen3vl"],
        )

    def test_unknown_model_raises_with_valid_options(self):
        with self.assertRaisesRegex(ValueError, "qwen3vl"):
            get_adapter("yolo9000")

    def test_every_adapter_provides_identity_and_config(self):
        for name in ADAPTERS:
            adapter = get_adapter(name)
            identity = adapter.identity()
            self.assertEqual(
                sorted(identity),
                ["model_name", "model_revision", "prompt_hash"],
                msg=name,
            )
            self.assertIsInstance(adapter.generation_config, dict, msg=name)


class QwenIdentityContinuityTests(unittest.TestCase):
    """Pin the historical identity values so run fingerprints never drift."""

    def test_model_constants_match_historical_config(self):
        self.assertEqual(MODEL_NAME, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(MODEL_REVISION, "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
        self.assertEqual(MIN_PIXELS, 256 * 28 * 28)
        self.assertEqual(MAX_PIXELS, 3072 * 28 * 28)

    def test_prompt_hash_matches_prompts_module(self):
        adapter = get_adapter("qwen3vl")
        self.assertEqual(
            adapter.prompt_hash(), grounding_prompt_hash(GROUNDING_SYSTEM_PROMPT)
        )

    def test_default_generation_config_is_backward_compatible(self):
        adapter = get_adapter("qwen3vl")
        self.assertEqual(
            adapter.generation_config,
            {"max_new_tokens": 32, "do_sample": False, "max_pixels": MAX_PIXELS},
        )


class Qwen3_8IdentityTests(unittest.TestCase):
    """Pin Qwen3.8-27B identity values so run fingerprints never drift."""

    def test_model_constants_match_adoption_pin(self):
        from aicomp_grounding.models.qwen3_8 import (
            MAX_PIXELS as Q38_MAX_PIXELS,
            MIN_PIXELS as Q38_MIN_PIXELS,
            MODEL_NAME as Q38_MODEL_NAME,
            MODEL_REVISION as Q38_MODEL_REVISION,
        )

        self.assertEqual(Q38_MODEL_NAME, "Qwen/Qwen3.8-27B")
        self.assertEqual(
            Q38_MODEL_REVISION, "e823e888ae179eb3be02c1a48899c4f828371376"
        )
        self.assertEqual(Q38_MIN_PIXELS, 256 * 28 * 28)
        self.assertEqual(Q38_MAX_PIXELS, 3072 * 28 * 28)

    def test_prompt_hash_matches_prompts_module(self):
        adapter = get_adapter("qwen3_8")
        self.assertEqual(
            adapter.prompt_hash(), grounding_prompt_hash(GROUNDING_SYSTEM_PROMPT)
        )

    def test_identity_fields_are_complete(self):
        adapter = get_adapter("qwen3_8")
        identity = adapter.identity()
        self.assertEqual(
            sorted(identity), ["model_name", "model_revision", "prompt_hash"]
        )
        self.assertEqual(identity["model_name"], "Qwen/Qwen3.8-27B")

    def test_training_hyperparameters_clone_qwen3vl_baseline(self):
        adapter = get_adapter("qwen3_8")
        hyper = adapter.training_hyperparameters()
        self.assertEqual(hyper["batch_size"], 1)
        self.assertEqual(hyper["gradient_accumulation_steps"], 16)
        self.assertEqual(hyper["learning_rate"], 1e-4)
        self.assertEqual(hyper["epochs"], 3)
        self.assertEqual(hyper["lora_rank"], 16)
        self.assertEqual(hyper["lora_alpha"], 48)
        self.assertEqual(hyper["compute_dtype"], "bfloat16")
        self.assertEqual(
            adapter.lora_target_modules(),
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )


class Qwen3_8ModelSourceTests(unittest.TestCase):
    def test_explicit_model_path_returns_local_source(self):
        from aicomp_grounding.models.qwen3_8 import _resolve_model_source

        self.assertEqual(
            _resolve_model_source("/tmp/qwen"),
            ("/tmp/qwen", False),
        )

    def test_automatic_source_uses_modelscope_revision(self):
        from unittest import mock

        from aicomp_grounding.models import qwen3_8

        fake_modelscope = types.ModuleType("modelscope")
        calls = []

        def snapshot_download(model_id, revision):
            calls.append((model_id, revision))
            return "/tmp/downloaded/qwen"

        fake_modelscope.snapshot_download = snapshot_download

        with mock.patch.dict(sys.modules, {"modelscope": fake_modelscope}):
            source, from_hub = qwen3_8._resolve_model_source(None)

        self.assertEqual(source, "/tmp/downloaded/qwen")
        self.assertFalse(from_hub)
        self.assertEqual(
            calls,
            [
                (
                    qwen3_8.MODEL_NAME,
                    qwen3_8.MODEL_REVISION,
                )
            ],
        )

    def test_automatic_source_without_modelscope_raises_actionable_error(self):
        from unittest import mock

        from aicomp_grounding.models.qwen3_8 import _resolve_model_source

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "modelscope":
                raise ImportError("No module named 'modelscope'")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "--model-path"):
                _resolve_model_source(None)


class Qwen3_8ChatTemplateTests(unittest.TestCase):
    """Qwen3.8 thinks by default; grounding must always disable it."""

    def test_chat_template_kwargs_disable_thinking(self):
        from aicomp_grounding.models.qwen3_8 import CHAT_TEMPLATE_KWARGS

        self.assertEqual(CHAT_TEMPLATE_KWARGS, {"enable_thinking": False})

    def test_apply_chat_template_helper_forwards_non_thinking_kwargs(self):
        from aicomp_grounding.models.qwen3_8 import _apply_chat_template

        class FakeProcessor:
            def __init__(self):
                self.kwargs = None

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt,
                **kwargs,
            ):
                self.kwargs = kwargs
                return "template-text"

        processor = FakeProcessor()
        text = _apply_chat_template(
            processor,
            [{"role": "user", "content": "Locate: the person"}],
            add_generation_prompt=True,
        )
        self.assertEqual(text, "template-text")
        self.assertEqual(processor.kwargs, {"enable_thinking": False})

    def test_inference_builders_use_non_thinking_chat_template(self):
        from PIL import Image

        from aicomp_grounding.models.base import ModelInput

        class FakeProcessor:
            def __init__(self):
                self.calls = []

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt,
                **kwargs,
            ):
                self.calls.append(kwargs)
                return "template-text"

            def __call__(self, **kwargs):
                return {"inputs_ready": True}

        fake_vision = types.ModuleType("qwen_vl_utils")
        fake_vision.process_vision_info = lambda messages: ([], [])
        image = Image.new("RGB", (8, 8))
        sample = ModelInput(visible=image, infrared=image, depth=image, query="the person")
        processor = FakeProcessor()
        adapter = get_adapter("qwen3_8")
        adapter._processor = processor

        with mock.patch.dict(sys.modules, {"qwen_vl_utils": fake_vision}):
            adapter.build_grounding_batch([sample], processor=processor)
            adapter.prepare_inputs([sample])

        self.assertTrue(processor.calls)
        self.assertTrue(
            all(call == {"enable_thinking": False} for call in processor.calls)
        )


class Qwen3_8CollateTests(unittest.TestCase):
    def test_collate_preserves_multimodal_token_types(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is required for tensor collation")

        class Tokenizer:
            pad_token_id = 0

        class Processor:
            tokenizer = Tokenizer()

        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([-100, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "pixel_values": torch.zeros((1, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
                "mm_token_type_ids": torch.tensor([0, 1]),
            },
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([-100, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "pixel_values": torch.zeros((1, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
                "mm_token_type_ids": torch.tensor([0, 1, 1]),
            },
        ]
        result = get_adapter("qwen3_8").collate_training_batch(
            batch,
            processor=Processor(),
        )
        self.assertEqual(
            result["mm_token_type_ids"].tolist(), [[0, 1, 0], [0, 1, 1]]
        )


class TrainableAdapterContractTests(unittest.TestCase):
    def test_vlm_adapters_expose_training_contract(self):
        for name in ("qwen3vl", "qwen3_8", "internvl35"):
            adapter = get_adapter(name)
            hyperparameters = adapter.training_hyperparameters()
            self.assertEqual(hyperparameters["lora_rank"], 16)
            self.assertEqual(hyperparameters["lora_alpha"], 48)
            self.assertIn("q_proj", adapter.lora_target_modules())


class InternVLContractTests(unittest.TestCase):
    def test_uses_real_max_patches_budget(self):
        adapter = get_adapter("internvl35")
        self.assertEqual(adapter.generation_config["max_patches"], 12)
        self.assertNotIn("max_num_tiles", adapter.generation_config)
        self.assertEqual(adapter.training_hyperparameters()["max_patches"], 12)
        with self.assertRaises(TypeError):
            get_adapter("internvl35", max_num_tiles=6)

    def test_question_has_numbered_context_image_slots_and_official_prompt(self):
        question = build_grounding_question("the red car")
        self.assertEqual(question.count(INTERNVL_IMAGE_TOKEN), 3)
        self.assertIn("Image-1: " + INTERNVL_IMAGE_TOKEN, question)
        self.assertIn("Image-3: " + INTERNVL_IMAGE_TOKEN, question)
        self.assertNotIn("<image>", question)
        self.assertIn("<ref>the red car</ref>", question)
        self.assertIn("visible RGB, infrared, and depth", question)

    def test_parse_box_with_tags_scales_from_1000(self):
        text = 'The target is <ref>the car</ref><box>[[100,200,300,400]]</box>.'
        self.assertEqual(
            parse_internvl_box(text), [0.1, 0.2, 0.3, 0.4]
        )

    def test_parse_bare_double_bracket_output(self):
        self.assertEqual(
            parse_internvl_box("[[250,250,750,750]]"), [0.25, 0.25, 0.75, 0.75]
        )

    def test_parse_rejects_ambiguous_or_invalid(self):
        self.assertIsNone(parse_internvl_box("no boxes here"))
        self.assertIsNone(
            parse_internvl_box("[[1,2,3,4]] and [[5,6,7,8]]")  # ambiguous
        )
        self.assertIsNone(parse_internvl_box("<box>[[900,0,100,100]]</box>"))  # x1>x2
        self.assertIsNone(parse_internvl_box("<box>[[500,500,500,900]]</box>"))  # zero width


class GroundingDINOContractTests(unittest.TestCase):
    def test_normalize_grounding_query_matches_official_demo(self):
        self.assertEqual(
            normalize_grounding_query(" The Red Car "),
            "the red car.",
        )
        self.assertEqual(
            normalize_grounding_query("the red car."),
            "the red car.",
        )

    def test_select_top_detection_picks_highest_postprocessed_xyxy(self):
        # post_process_grounded_object_detection already returns normalized XYXY.
        boxes = [[0.5, 0.5, 0.75, 0.75], [0.25, 0.25, 0.375, 0.375]]
        bbox, score = select_top_detection(boxes, [0.3, 0.9])
        self.assertEqual(bbox, [0.25, 0.25, 0.375, 0.375])
        self.assertEqual(score, 0.9)

    def test_select_top_detection_falls_back_to_next_valid_box(self):
        boxes = [[0.8, 0.8, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]]
        bbox, score = select_top_detection(boxes, [0.9, 0.6])
        self.assertEqual(bbox, [0.1, 0.1, 0.5, 0.5])
        self.assertEqual(score, 0.6)

    def test_select_top_detection_rejects_invalid_or_low_score(self):
        self.assertEqual(
            select_top_detection([[0.8, 0.8, 0.4, 0.4]], [0.9]),
            (None, None),
        )
        self.assertEqual(
            select_top_detection([[0.1, 0.1, 0.5, 0.5]], [0.05]), (None, None)
        )
        self.assertEqual(select_top_detection([], []), (None, None))


class MockAdapterTests(unittest.TestCase):
    def test_predictions_are_deterministic_and_valid(self):
        from aicomp_grounding.bbox import validate_bbox

        self.assertEqual(_stable_box("q"), _stable_box("q"))
        self.assertNotEqual(_stable_box("q"), _stable_box("other"))
        self.assertIsNotNone(validate_bbox(_stable_box("q")))

    def test_mock_predict_returns_scores(self):
        adapter = get_adapter("mock")
        adapter.load()
        samples = [
            ModelInput(visible=None, infrared=None, depth=None, query="a", key="a"),
            ModelInput(visible=None, infrared=None, depth=None, query="b", key="b"),
        ]
        results = adapter.predict(samples)
        self.assertEqual([r.score for r in results], [0.5, 0.5])
        self.assertTrue(all(r.bbox is not None for r in results))


class MockPipelineEndToEndTests(unittest.TestCase):
    """Full inference chain on CPU: items -> mock adapter -> metrics -> ZIP."""

    @staticmethod
    def _template(count: int = 3) -> dict:
        return {
            f"{index:06d}_001": {
                "visible": f"Images/visible/{index:06d}.png",
                "infrared": f"Images/infrared/{index:06d}.png",
                "depth": f"Images/depth/{index:06d}.png",
                "query": f"The target object number {index}",
            }
            for index in range(1, count + 1)
        }

    def test_items_to_submission_zip(self):
        adapter = get_adapter("mock")
        adapter.load()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index_path = root / "test.json"
            atomic_write_json(index_path, self._template())
            items, metadata = load_inference_items(index_path)
            self.assertIsNone(metadata)

            samples = []
            predictions = {}
            for item in items:
                samples.append(
                    ModelInput(
                        visible=None,
                        infrared=None,
                        depth=None,
                        query=item["query"],
                        key=item["key"],
                    )
                )
            for sample, result in zip(samples, adapter.predict(samples)):
                predictions[sample.key] = result.bbox

            self.assertIsNone(evaluate_predictions(items, predictions))

            predictions_path = root / "predictions.json"
            atomic_write_json(predictions_path, predictions)
            zip_path = build_submission(
                test_json_path=index_path,
                predictions_path=predictions_path,
                output_dir=root / "out",
                expected_query_count=None,
                expected_template_sha256=None,
            )
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["result.json"])
                payload = json.loads(archive.read("result.json"))
            self.assertEqual(len(payload), 3)
            for entry in payload.values():
                self.assertIsNotNone(entry["bbox"])


if __name__ == "__main__":
    unittest.main()
