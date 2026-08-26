import json
import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.query_style import (
    QUERY_STYLE_GROUPS,
    analyze_queries,
    build_style_plan,
    build_style_prompt,
    classify_query,
    expand_annotation_source,
    extract_supported_styles,
    parse_scene_card,
    style_plan_fingerprint,
)
from scripts.build_style_plan import build_style_plan_artifacts


def _dataset() -> dict:
    return {
        "001_00000001": {
            "visible": "Train/001/color/00000001.png",
            "infrared": "Train/001/infrared/00000001.png",
            "depth": "Processed/Train/001/depth_jet/00000001.png",
            "bbox": [0.1, 0.1, 0.3, 0.3],
            "width": 1920,
            "height": 1080,
            "query": "",
        }
    }


class QueryStyleTests(unittest.TestCase):
    def test_classify_query_covers_all_groups(self):
        self.assertEqual(classify_query("The third cone from left to right"), "ordinal")
        self.assertEqual(classify_query("The car to the right of the van"), "spatial_landmark")
        self.assertEqual(classify_query("The farthest drone from the camera"), "distance")
        self.assertEqual(classify_query("The deer in the middle of the field"), "scene_location")
        self.assertEqual(classify_query("The person wearing a red jacket"), "attribute_action")

    def test_analyze_queries_returns_full_semantic_groups(self):
        queries = [
            "The third traffic cone",
            "The car beside the wall",
            "The deer in the middle of the field",
            "The person wearing a blue shirt",
        ]
        stats = analyze_queries(queries)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(set(stats["group_counts"]), set(QUERY_STYLE_GROUPS))
        self.assertEqual(stats["group_counts"]["ordinal"], 1)
        self.assertEqual(stats["mean_words"], 5.75)
        self.assertAlmostEqual(stats["group_ratios"]["ordinal"], 0.25)

    def test_scene_card_parser_and_supported_styles(self):
        card = parse_scene_card(
            json.dumps(
                {
                    "scene_type": "plaza",
                    "tracked_category": "person",
                    "same_category_count": 3,
                    "ordinal_position": "third from left to right",
                    "stable_attributes": ["red jacket"],
                    "action_or_state": "sitting",
                    "landmark": "stone ledge",
                    "scene_region": "right side of the field",
                    "distance_hint": "foreground",
                    "unique_in_scene": False,
                }
            )
        )
        supported = extract_supported_styles(card)
        self.assertIn("ordinal", supported)
        self.assertIn("spatial_landmark", supported)
        self.assertIn("distance", supported)
        self.assertIn("scene_location", supported)
        self.assertIn("attribute_action", supported)

    def test_scene_card_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            parse_scene_card('{"scene_type":"plaza"}')

    def test_style_plan_expands_synthetic_sample_ids(self):
        dataset = _dataset()
        plan = build_style_plan(
            dataset,
            {
                "001": {
                    "scene_type": "plaza",
                    "tracked_category": "cone",
                    "same_category_count": 3,
                    "ordinal_position": "third",
                    "stable_attributes": [],
                    "action_or_state": None,
                    "landmark": "barrier",
                    "scene_region": "front row",
                    "distance_hint": "foreground",
                    "unique_in_scene": False,
                }
            },
            seed=42,
            queries_per_frame=2,
        )
        expanded = expand_annotation_source(dataset, plan)
        self.assertEqual(len(plan["items"]), 2)
        self.assertEqual(len(expanded), 2)
        for key in expanded:
            self.assertTrue(key.startswith("001_00000001_q"))
            self.assertEqual(expanded[key]["bbox"], dataset["001_00000001"]["bbox"])
            self.assertIn("annotation_style", expanded[key])
        self.assertTrue(style_plan_fingerprint(plan))

    def test_style_prompt_contains_requested_family_and_fallback(self):
        prompt = build_style_prompt(
            "ordinal",
            min_words=8,
            max_words=12,
            template_family="FROM_LEFT_TO_RIGHT",
            fallback_style="scene_location",
        )
        self.assertIn("FROM_LEFT_TO_RIGHT", prompt)
        self.assertIn("scene_location", prompt)
        self.assertIn("never mention the rectangle", prompt)


class BuildStylePlanArtifactsTests(unittest.TestCase):
    def test_artifacts_are_generation_compatible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / "data"
            (data_root / "Train").mkdir(parents=True)
            (data_root / "Processed").mkdir(parents=True)
            dataset = _dataset()
            atomic_write_json(data_root / "train.json", dataset)
            atomic_write_json(
                data_root / "split_manifest.json",
                {
                    "status": "complete",
                    "preparation_protocol_version": 2,
                    "index_fingerprints": {"train": "raw"},
                    "index_sample_counts": {"train": 1},
                    "stats": {"train_samples": 1, "val_samples": 0},
                },
            )
            cards_path = root / "cards.json"
            atomic_write_json(
                cards_path,
                {
                    "001": {
                        "scene_type": "field",
                        "tracked_category": "deer",
                        "same_category_count": 1,
                        "ordinal_position": None,
                        "stable_attributes": [],
                        "action_or_state": "standing",
                        "landmark": "rock",
                        "scene_region": "middle of the field",
                        "distance_hint": None,
                        "unique_in_scene": True,
                    }
                },
            )
            result = build_style_plan_artifacts(
                split="train",
                data_root=data_root,
                scene_cards_path=cards_path,
                output_root=root / "outputs",
                queries_per_frame=2,
            )
            source = json.loads(
                (Path(result["data_root_for_generation"]) / "train.json").read_text()
            )
            manifest = json.loads(
                (Path(result["data_root_for_generation"]) / "split_manifest.json").read_text()
            )
            self.assertEqual(result["expanded_samples"], 2)
            self.assertEqual(len(source), 2)
            self.assertEqual(manifest["index_sample_counts"]["train"], 2)
            self.assertTrue(manifest["expanded_style_plan_fingerprint"])
            self.assertTrue((Path(result["data_root_for_generation"]) / "Train").is_symlink())


if __name__ == "__main__":
    unittest.main()
