import unittest

from aicomp_grounding.query import clean_query_text, validate_generated_query


class QueryValidationTests(unittest.TestCase):
    def test_accepts_concise_english_description(self):
        valid, reason = validate_generated_query(
            "A person in a black jacket standing beside the left wall."
        )
        self.assertTrue(valid, reason)

    def test_rejects_annotation_scaffolding_and_coordinates(self):
        self.assertFalse(validate_generated_query("The object inside the red box near the wall.")[0])
        self.assertFalse(validate_generated_query("The target is at (100, 200), near the wall.")[0])
        self.assertFalse(
            validate_generated_query(
                "The pedestrian wearing <|image_pad|> a bright yellow jacket near the road"
            )[0]
        )

    def test_cleans_wrapping_markup(self):
        self.assertEqual(clean_query_text('`"The small red chair beside the wooden desk."`'), "The small red chair beside the wooden desk.")


if __name__ == "__main__":
    unittest.main()
