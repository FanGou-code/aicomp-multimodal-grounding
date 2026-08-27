import unittest

from aicomp_grounding.query import (
    clean_query_text,
    validate_generated_query,
    validate_query_style,
)


class QueryValidationTests(unittest.TestCase):
    def test_accepts_concise_english_description(self):
        valid, reason = validate_generated_query(
            "A person in a black jacket standing beside the left wall."
        )
        self.assertTrue(valid, reason)

    def test_rejects_annotation_scaffolding_and_coordinates(self):
        self.assertFalse(validate_generated_query("The object inside the red rectangle near the wall.")[0])
        self.assertFalse(validate_generated_query("The target is at (100, 200), near the wall.")[0])
        # "red box" is a legitimate description of an object shape; only
        # explicit annotation-marking language is rejected.
        self.assertTrue(validate_generated_query("The red box with a yellow top.")[0])

    def test_cleans_wrapping_markup(self):
        self.assertEqual(
            clean_query_text('`"The small red chair beside the wooden desk."`'),
            "The small red chair beside the wooden desk",
        )


class QueryStyleTests(unittest.TestCase):
    def test_rejects_short_bare_label(self):
        valid, _ = validate_query_style("White hat")
        self.assertFalse(valid)

    def test_accepts_short_query_with_spatial_word(self):
        valid, _ = validate_query_style("Leftmost white cone")
        self.assertTrue(valid)

    def test_accepts_long_query_without_spatial_word(self):
        valid, _ = validate_query_style("The red sedan with a dent parked on the road")
        self.assertTrue(valid)

    def test_rejects_empty_query(self):
        valid, reason = validate_query_style("")
        self.assertFalse(valid)
        self.assertIn("empty", reason)

    def test_accepts_short_query_with_ordinal(self):
        valid, _ = validate_query_style("Third cone in the row")
        self.assertTrue(valid)

    def test_rejects_short_query_without_multi_object_cue(self):
        # 4 words, below threshold, no spatial/ordinal cue
        valid, _ = validate_query_style("The bird standing alone")
        self.assertFalse(valid)

    def test_accepts_short_query_with_comparison_cue(self):
        valid, _ = validate_query_style("The larger white umbrella")
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
