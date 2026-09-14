"""Adapter contract tests + torch-free end-to-end pipeline test (mock model)."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path

from aicomp_grounding.inference_core import evaluate_predictions, load_inference_items
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.models import ADAPTERS, available_models, get_adapter
from aicomp_grounding.models.base import (
    DEFAULT_LORA_PROJECTIONS,
    ModelInput,
    language_model_lora_targets,
)
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
from aicomp_grounding.models.glm46v import (
    GLM_BOX_CLOSE as GLM46V_BOX_CLOSE,
    GLM_BOX_OPEN as GLM46V_BOX_OPEN,
    MAX_PIXELS as GLM46V_MAX_PIXELS,
    MODEL_NAME as GLM46V_MODEL_NAME,
    MODEL_REVISION as GLM46V_MODEL_REVISION,
    MODELSCOPE_NAME as GLM46V_MODELSCOPE_NAME,
    format_glm_bbox,
    parse_glm_box,
)
from aicomp_grounding.models.qwen3_5 import (
    MAX_PIXELS as QWEN3_5_MAX_PIXELS,
    MIN_PIXELS as QWEN3_5_MIN_PIXELS,
    MODEL_NAME as QWEN3_5_MODEL_NAME,
    MODEL_REVISION as QWEN3_5_MODEL_REVISION,
)
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
            ["glm46v", "groundingdino", "internvl35", "mimo_vl", "mock", "qwen36_27b", "qwen3_5", "qwen3vl"],
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


class Qwen3_5IdentityContinuityTests(unittest.TestCase):
    """Pin the Qwen3.5-9B identity values so run fingerprints never drift."""

    def test_model_constants_match_pinned_config(self):
        self.assertEqual(QWEN3_5_MODEL_NAME, "Qwen/Qwen3.5-9B")
        self.assertEqual(
            QWEN3_5_MODEL_REVISION, "460979c3d11864dd16408d860ac930a360a2fac2"
        )
        self.assertEqual(QWEN3_5_MIN_PIXELS, 256 * 28 * 28)
        self.assertEqual(QWEN3_5_MAX_PIXELS, 3072 * 28 * 28)

    def test_prompt_hash_matches_prompts_module(self):
        adapter = get_adapter("qwen3_5")
        self.assertEqual(
            adapter.prompt_hash(), grounding_prompt_hash(GROUNDING_SYSTEM_PROMPT)
        )

    def test_default_generation_config_is_backward_compatible(self):
        adapter = get_adapter("qwen3_5")
        self.assertEqual(
            adapter.generation_config,
            {
                "max_new_tokens": 32,
                "do_sample": False,
                "max_pixels": QWEN3_5_MAX_PIXELS,
            },
        )

    def test_thinking_disabled_in_chat_template_kwargs(self):
        # The grounding protocol must disable Qwen3.5 thinking for both
        # training and inference; the kwarg is applied via _apply_chat_template.
        from aicomp_grounding.models import qwen3_5

        self.assertEqual(
            qwen3_5.CHAT_TEMPLATE_KWARGS, {"enable_thinking": False}
        )


class Glm46VContractTests(unittest.TestCase):
    def test_invalid_numbers_are_not_parsed_as_positive_substrings(self):
        for body in ("-100,200,300,400", "100.5,200,300,400", "1,2,3,4,5"):
            with self.subTest(body=body):
                self.assertIsNone(parse_glm_box(body))
                self.assertIsNone(parse_glm_box(GLM46V_BOX_OPEN + body + GLM46V_BOX_CLOSE))

    def test_quantized_edge_box_is_valid(self):
        self.assertEqual(
            parse_glm_box(format_glm_bbox([0.9996, 0.1, 1, 0.5])),
            [0.999, 0.1, 1.0, 0.5],
        )

    def test_valid_bare_box_with_prose_or_incomplete_delimiter_stays_supported(self):
        for text in ("Box: 100,200,300,400.", GLM46V_BOX_OPEN + "100,200,300,400"):
            with self.subTest(text=text):
                self.assertEqual(parse_glm_box(text), [0.1, 0.2, 0.3, 0.4])

    """Pin GLM-4.6V-Flash identity, box contract, and thinking-disabled kwarg."""

    def test_model_constants_match_pinned_config(self):
        self.assertEqual(GLM46V_MODEL_NAME, "zai-org/GLM-4.6V-Flash")
        self.assertEqual(
            GLM46V_MODEL_REVISION, "a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41"
        )
        # ModelScope hosts GLM under ZhipuAI (not zai-org); the auto-load
        # fallback must use the ModelScope org, not MODEL_NAME.
        self.assertEqual(GLM46V_MODELSCOPE_NAME, "ZhipuAI/GLM-4.6V-Flash")

    def test_box_tokens_are_the_glm_added_vocab_delimiters(self):
        self.assertEqual(GLM46V_BOX_OPEN, chr(0x3C) + "|begin_of_box|" + chr(0x3E))
        self.assertEqual(GLM46V_BOX_CLOSE, chr(0x3C) + "|end_of_box|" + chr(0x3E))

    def test_default_generation_config_is_backward_compatible(self):
        # GLM-4.6V's identity moved once, deliberately: the adapter now pins the
        # shared pixel budget instead of inheriting the checkpoint's
        # `longest_edge`, so `max_pixels` enters its run identity.  Any run of
        # this adapter needs a new tag.  At 1920x1080 the grid is unchanged.
        adapter = get_adapter("glm46v")
        self.assertEqual(
            adapter.generation_config,
            {"max_new_tokens": 32, "do_sample": False, "max_pixels": GLM46V_MAX_PIXELS},
        )

    def test_thinking_disabled_in_chat_template_kwargs(self):
        from aicomp_grounding.models import glm46v

        self.assertEqual(glm46v.CHAT_TEMPLATE_KWARGS, {"enable_thinking": False})

    def test_format_and_parse_round_trip(self):
        text = format_glm_bbox([0.1, 0.2, 0.3, 0.4])
        expected = GLM46V_BOX_OPEN + "100,200,300,400" + GLM46V_BOX_CLOSE
        self.assertEqual(text, expected)
        self.assertEqual(parse_glm_box(text), [0.1, 0.2, 0.3, 0.4])

    def test_parse_tolerates_surrounding_prose_and_bare_ints(self):
        tagged = (
            "The target is "
            + GLM46V_BOX_OPEN
            + "100,200,300,400"
            + GLM46V_BOX_CLOSE
            + "."
        )
        self.assertEqual(parse_glm_box(tagged), [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(parse_glm_box("100,200,300,400"), [0.1, 0.2, 0.3, 0.4])

    def test_parse_rejects_ambiguous_or_invalid(self):
        two = (
            GLM46V_BOX_OPEN
            + "1,2,3,4"
            + GLM46V_BOX_CLOSE
            + " and "
            + GLM46V_BOX_OPEN
            + "5,6,7,8"
            + GLM46V_BOX_CLOSE
        )
        self.assertIsNone(parse_glm_box(two))
        self.assertIsNone(parse_glm_box(GLM46V_BOX_OPEN + "500,500,500,900" + GLM46V_BOX_CLOSE))
        self.assertIsNone(parse_glm_box(GLM46V_BOX_OPEN + "100,200,100,400" + GLM46V_BOX_CLOSE))
        self.assertIsNone(parse_glm_box(GLM46V_BOX_OPEN + "0,0,2000,400" + GLM46V_BOX_CLOSE))
        self.assertIsNone(parse_glm_box(""))
        self.assertIsNone(parse_glm_box("no box here"))


TRAINABLE_ADAPTERS = (
    "qwen3vl",
    "qwen3_5",
    "qwen36_27b",
    "mimo_vl",
    "glm46v",
    "internvl35",
)

#: Projections each adapter's language model exposes, and therefore the exact
#: target set it must declare.  `glm46v` fuses the MLP gate and up projections
#: into `gate_up_proj`, so the split pair does not apply to it and it trains no
#: up path at all; widening that adapter is a recipe change, so its set is pinned
#: here rather than assumed.
EXPECTED_LORA_PROJECTIONS = {
    name: DEFAULT_LORA_PROJECTIONS
    for name in ("qwen3vl", "qwen3_5", "qwen36_27b", "mimo_vl", "internvl35")
}
EXPECTED_LORA_PROJECTIONS["glm46v"] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "down_proj",
)

# Vision towers that reuse language-model projection names are the trap: a bare
# suffix list silently wraps them, which puts the vision encoder through
# backward plus gradient-checkpoint recomputation.
FROZEN_VISION_MODULES = (
    "model.visual.blocks.0.mlp.gate_proj",
    "model.visual.blocks.0.mlp.up_proj",
    "model.visual.blocks.0.mlp.down_proj",
    "model.visual.merger.gate_proj",
    "model.visual.merger.up_proj",
    "model.visual.merger.down_proj",
    "model.vision_tower.encoder.layers.0.attention.q_proj",
    "model.vision_tower.encoder.layers.0.attention.k_proj",
    "model.vision_tower.encoder.layers.0.attention.v_proj",
)


class TrainableAdapterContractTests(unittest.TestCase):
    def test_vlm_adapters_expose_training_contract(self):
        for name in TRAINABLE_ADAPTERS:
            adapter = get_adapter(name)
            hyperparameters = adapter.training_hyperparameters()
            self.assertEqual(hyperparameters["lora_rank"], 16)
            self.assertEqual(hyperparameters["lora_alpha"], 32)
            self.assertEqual(
                hyperparameters["lora_targets"], adapter.lora_target_modules()
            )

    def test_lora_targets_pin_the_declared_projection_set(self):
        """Anchoring is shared; the projection set is each adapter's own."""
        for name in TRAINABLE_ADAPTERS:
            self.assertEqual(
                get_adapter(name).lora_target_modules(),
                language_model_lora_targets(*EXPECTED_LORA_PROJECTIONS[name]),
                f"{name} changed its LoRA scope",
            )

    def test_lora_targets_never_reach_the_vision_tower(self):
        """The encoder stays frozen whichever projections an adapter declares.

        PEFT matches a string ``target_modules`` with ``re.fullmatch``, so this is
        the same test PEFT applies when injecting the adapter.
        """
        for name in TRAINABLE_ADAPTERS:
            pattern = get_adapter(name).lora_target_modules()
            self.assertIsInstance(pattern, str, f"{name} must return a regex, not a list")
            for key in FROZEN_VISION_MODULES:
                self.assertIsNone(re.fullmatch(pattern, key), f"{name} would train {key}")


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


class LocalModelPolicyTests(unittest.TestCase):
    def test_real_adapters_reject_missing_directory_before_loading(self):
        for name in ("qwen3vl", "qwen3_5", "glm46v", "internvl35", "groundingdino"):
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, "--model-path"):
                    get_adapter(name).load()
                with tempfile.TemporaryDirectory() as td:
                    with self.assertRaises(FileNotFoundError):
                        get_adapter(name).load(model_path=td)

    def test_loaders_explicitly_forbid_network_fallback(self):
        import ast
        import inspect
        for name in ("qwen3vl", "qwen3_5", "glm46v", "internvl35", "groundingdino"):
            module = inspect.getmodule(type(get_adapter(name)))
            calls = [n for n in ast.walk(ast.parse(inspect.getsource(module)))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "from_pretrained"]
            self.assertTrue(calls, name)
            for call in calls:
                self.assertTrue(any(k.arg == "local_files_only" and isinstance(k.value, ast.Constant)
                                    and k.value.value is True for k in call.keywords), name)

    def test_dino_instance_threshold_reaches_final_selection(self):
        import torch
        from transformers import BatchEncoding
        class Processor:
            def __call__(self, **kwargs):
                return BatchEncoding({"input_ids": torch.tensor([[1]])})
            def post_process_grounded_object_detection(self, outputs, ids, threshold, text_threshold):
                self.threshold = threshold
                return [{"boxes": torch.tensor([[0.1, 0.2, 0.5, 0.6]]), "scores": torch.tensor([0.2])}]
        class Model:
            device = "cpu"
            def __call__(self, **kwargs):
                return None
        adapter = get_adapter("groundingdino", box_threshold=0.1)
        adapter._processor, adapter._model = Processor(), Model()
        result = adapter.predict([ModelInput(None, None, None, "the object")])[0]
        self.assertEqual(adapter._processor.threshold, 0.1)
        self.assertIsNotNone(result.bbox)
        self.assertAlmostEqual(result.score, 0.2)
        self.assertEqual(adapter.compute_dtype, "float32")


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
