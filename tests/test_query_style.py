import unittest

from aicomp_grounding.query_style import (
    DISAMBIGUATION_PROMPT_HASH,
    DISAMBIGUATION_QUERY_PROMPT,
    QUERY_STYLE_GROUPS,
    STYLE_PROMPT_HASH,
    analyze_queries,
    classify_query,
    normalize_style,
)


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

    def test_normalize_style(self):
        self.assertEqual(normalize_style("ordinal"), "ordinal")
        self.assertEqual(normalize_style("spatial"), "spatial_landmark")
        self.assertEqual(normalize_style("landmark"), "spatial_landmark")
        self.assertEqual(normalize_style("attribute"), "attribute_action")

    def test_disambiguation_prompt_invariants(self):
        self.assertTrue(DISAMBIGUATION_PROMPT_HASH)
        self.assertEqual(DISAMBIGUATION_PROMPT_HASH, STYLE_PROMPT_HASH)
        self.assertIn("target_category", DISAMBIGUATION_QUERY_PROMPT)
        self.assertIn("disambiguation_cue", DISAMBIGUATION_QUERY_PROMPT)
        self.assertIn("final_query", DISAMBIGUATION_QUERY_PROMPT)
        self.assertIn("viewer", DISAMBIGUATION_QUERY_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
