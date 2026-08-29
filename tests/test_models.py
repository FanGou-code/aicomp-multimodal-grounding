"""Adapter contract tests + torch-free end-to-end pipeline test (mock model)."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

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
            ["groundingdino", "internvl35", "mock", "qwen3vl"],
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


class TrainableAdapterContractTests(unittest.TestCase):
    def test_vlm_adapters_expose_training_contract(self):
        for name in ("qwen3vl", "internvl35"):
            adapter = get_adapter(name)
            hyperparameters = adapter.training_hyperparameters()
            self.assertEqual(hyperparameters["lora_rank"], 16)
            self.assertEqual(hyperparameters["lora_alpha"], 48)
            self.assertIn("q_proj", adapter.lora_target_modules())


class InternVLContractTests(unittest.TestCase):
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
