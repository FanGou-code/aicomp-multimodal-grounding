import unittest

from aicomp_grounding.query import (
    clean_query_text,
    validate_generated_query,
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

    def test_flip_spatial_query(self):
        from aicomp_grounding.query import flip_spatial_query

        flipped, changed = flip_spatial_query("the person on the left side")
        self.assertTrue(changed)
        self.assertEqual(flipped, "the person on the right side")

        flipped, changed = flip_spatial_query("the leftmost box and the rightmost chair")
        self.assertTrue(changed)
        self.assertEqual(flipped, "the rightmost box and the leftmost chair")

        flipped, changed = flip_spatial_query("Left corner to RIGHT edge")
        self.assertTrue(changed)
        self.assertEqual(flipped, "Right corner to LEFT edge")

        flipped, changed = flip_spatial_query("the red apple on the table")
        self.assertFalse(changed)
        self.assertEqual(flipped, "the red apple on the table")


if __name__ == "__main__":
    unittest.main()
