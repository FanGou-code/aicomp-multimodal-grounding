"""Adapter contract tests + torch-free end-to-end pipeline test (mock model)."""

from __future__ import annotations

import inspect
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
    ModelInput,
    language_model_lora_targets,
)
from aicomp_grounding.models.mock import _stable_box
from aicomp_grounding.models.glm46v import (
    GLM_BOX_CLOSE as GLM46V_BOX_CLOSE,
    GLM_BOX_OPEN as GLM46V_BOX_OPEN,
    MAX_PIXELS as GLM46V_MAX_PIXELS,
    MIN_PIXELS as GLM46V_MIN_PIXELS,
    MODEL_NAME as GLM46V_MODEL_NAME,
    MODEL_REVISION as GLM46V_MODEL_REVISION,
    MODELSCOPE_NAME as GLM46V_MODELSCOPE_NAME,
    format_glm_bbox,
    parse_glm_box,
    processor_pixel_kwargs,
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
            ["glm46v", "mock", "qwen3_5", "qwen3vl"],
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


#: The snapshot each adapter's weights are downloaded from; the same values are
#: listed in the README weight table, so the identity string and the bytes on
#: disk name the same revision.  A branch name would leave the string unchanged
#: while the weights underneath move.
EXPECTED_MODEL_REVISIONS = {
    "qwen3vl": "5d854aab08710c16b980ec6d603d863b3821b915",
    "qwen3_5": "460979c3d11864dd16408d860ac930a360a2fac2",
    "glm46v": "a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41",
}


class ModelRevisionPinTests(unittest.TestCase):
    """Every weighted adapter names a snapshot, not a branch."""

    def test_every_adapter_pins_a_snapshot_id(self):
        for name, revision in EXPECTED_MODEL_REVISIONS.items():
            with self.subTest(model=name):
                self.assertEqual(get_adapter(name).model_revision, revision)
                self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_revision_table_covers_every_weighted_adapter(self):
        self.assertEqual(sorted([*EXPECTED_MODEL_REVISIONS, "mock"]), available_models())


class QwenIdentityContinuityTests(unittest.TestCase):
    """Pin the historical identity values so run fingerprints never drift."""

    def test_model_constants_match_historical_config(self):
        self.assertEqual(MODEL_NAME, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(MODEL_REVISION, "5d854aab08710c16b980ec6d603d863b3821b915")
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
        # shared per-frame pixel budget instead of inheriting the checkpoint's
        # `longest_edge`, so `max_pixels` enters its run identity.  Any run of
        # this adapter needs a new tag.  Every image in the dataset is
        # 1920x1080, where the grid is 78x138 under either budget.
        adapter = get_adapter("glm46v")
        self.assertEqual(
            adapter.generation_config,
            {"max_new_tokens": 32, "do_sample": False, "max_pixels": GLM46V_MAX_PIXELS},
        )

    def test_thinking_disabled_in_chat_template_kwargs(self):
        from aicomp_grounding.models import glm46v

        self.assertEqual(glm46v.CHAT_TEMPLATE_KWARGS, {"enable_thinking": False})

    def test_processor_budget_converts_from_the_per_frame_unit(self):
        # Glm46VImageProcessor reads its budget from a `size` dict and compares
        # `temporal_factor * h * w` against `longest_edge`, so the per-frame
        # value the roster records is doubled exactly once, here.  Passing
        # `min_pixels` / `max_pixels` instead is silently dropped by the
        # processor and would leave the budget unpinned.
        self.assertEqual(
            processor_pixel_kwargs(GLM46V_MAX_PIXELS),
            {
                "size": {
                    "shortest_edge": 2 * GLM46V_MIN_PIXELS,
                    "longest_edge": 2 * GLM46V_MAX_PIXELS,
                }
            },
        )

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
    "glm46v",
)

#: Language-model projection names each adapter must declare, spelled out
#: literally.  Referring to the production constant here would move both sides
#: of the assertion at once, so deleting a projection from the shared default
#: would keep this test green -- the list is duplicated on purpose.
#: `glm46v` fuses the MLP gate and up projections into `gate_up_proj`, so the
#: split pair does not apply to it and it trains no up path at all; widening
#: that adapter is a recipe change, not a fix.
EXPECTED_LORA_PROJECTIONS = {
    "qwen3vl": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "qwen3_5": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "glm46v": ("q_proj", "k_proj", "v_proj", "o_proj", "down_proj"),
}

# Vision towers that reuse language-model projection names are the trap: a bare
# suffix list silently wraps them, which puts the vision encoder through
# backward plus gradient-checkpoint recomputation.  These are the real module
# paths of the GLM-4.6V vision tower (`glm46v`, in service): its blocks and its
# merger both expose `gate_proj` / `up_proj` / `down_proj`, so a de-anchored
# LoRA regex would wrap them.  The Qwen-family towers use other names
# (`linear_fc1` / `linear_fc2`, packed `qkv`), which is why they were never hit.
FROZEN_VISION_MODULES = (
    "model.visual.blocks.0.mlp.gate_proj",
    "model.visual.blocks.0.mlp.up_proj",
    "model.visual.blocks.0.mlp.down_proj",
    "model.visual.merger.gate_proj",
    "model.visual.merger.up_proj",
    "model.visual.merger.down_proj",
)

#: Where each projection lives in a language-model block, for the positive half
#: of the anchor assertion below.  Every tri-modal language model in the roster
#: nests attention under `self_attn` and the MLP under `mlp`.
PROJECTION_PARENT = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}


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

    def test_lora_targets_reach_the_language_model(self):
        """The anchor itself: every declared projection must match.

        The negative assertions below pass whether or not the pattern is
        anchored -- a bare alternation of projection names does not match a
        fully qualified path either.  This positive case is what keeps
        `model.language_model` in the pattern.
        """
        for name, projections in EXPECTED_LORA_PROJECTIONS.items():
            pattern = get_adapter(name).lora_target_modules()
            for projection in projections:
                key = (
                    f"model.language_model.layers.0."
                    f"{PROJECTION_PARENT[projection]}.{projection}"
                )
                with self.subTest(model=name, projection=projection):
                    self.assertIsNotNone(
                        re.fullmatch(pattern, key), f"{name} would not train {key}"
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


class OrdinalGenerationContractTests(unittest.TestCase):
    """Every adapter must be able to serve the ordinal module's generation call."""

    def test_every_registered_adapter_exposes_generate_messages(self):
        for name in available_models():
            with self.subTest(name=name):
                self.assertTrue(
                    callable(getattr(get_adapter(name), "generate_messages", None)),
                    f"{name} cannot serve the ordinal module",
                )

    def test_generate_messages_takes_keyword_only_generation_controls(self):
        parameters = inspect.signature(
            get_adapter("mock").generate_messages
        ).parameters
        for name in ("max_new_tokens", "temperature", "skip_special_tokens"):
            with self.subTest(parameter=name):
                self.assertEqual(
                    parameters[name].kind, inspect.Parameter.KEYWORD_ONLY
                )
        self.assertIs(parameters["skip_special_tokens"].default, False)


class LocalModelPolicyTests(unittest.TestCase):
    def test_real_adapters_reject_missing_directory_before_loading(self):
        for name in ("qwen3vl", "qwen3_5", "glm46v"):
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, "--model-path"):
                    get_adapter(name).load()
                with tempfile.TemporaryDirectory() as td:
                    with self.assertRaises(FileNotFoundError):
                        get_adapter(name).load(model_path=td)

    def test_loaders_explicitly_forbid_network_fallback(self):
        import ast
        import inspect
        for name in ("qwen3vl", "qwen3_5", "glm46v"):
            module = inspect.getmodule(type(get_adapter(name)))
            calls = [n for n in ast.walk(ast.parse(inspect.getsource(module)))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "from_pretrained"]
            self.assertTrue(calls, name)
            for call in calls:
                self.assertTrue(any(k.arg == "local_files_only" and isinstance(k.value, ast.Constant)
                                    and k.value.value is True for k in call.keywords), name)


class MockAdapterTests(unittest.TestCase):
    def test_predictions_are_deterministic_and_valid(self):
        from aicomp_grounding.bbox import validate_bbox

        self.assertEqual(_stable_box("q"), [0.272745, 0.133137, 0.612941, 0.484118])
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
