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

    def test_flip_refuses_glyph_order(self):
        """Mirroring text reverses the glyphs, so lateral order of letters is not image-plane order."""
        from aicomp_grounding.query import flip_spatial_query

        for query in (
            "The fourth letter from left to right",
            "The first letter from right to left",
            "The tallest word from the right",
            "The white symbol on the right side of the building",
        ):
            flipped, changed = flip_spatial_query(query)
            self.assertFalse(changed, query)
            self.assertEqual(flipped, query)

    def test_flip_refuses_egocentric_body_parts(self):
        """The person's own left/right is not the viewer's, so the mirrored caption would be false."""
        from aicomp_grounding.query import flip_spatial_query

        query = "The black and white shoe on the right foot of the person on the right"
        flipped, changed = flip_spatial_query(query)
        self.assertFalse(changed)
        self.assertEqual(flipped, query)

    def test_flip_allows_image_plane_laterality(self):
        """Ordering of objects on the image plane mirrors together with the pixels."""
        from aicomp_grounding.query import flip_spatial_query

        cases = {
            "The second flower from the left": "The second flower from the right",
            "The leftmost tree": "The rightmost tree",
            "The person on the far left": "The person on the far right",
            "The first person walking from left to right": "The first person walking from right to left",
            "The building on the left side of the image": "The building on the right side of the image",
        }
        for source, expected in cases.items():
            flipped, changed = flip_spatial_query(source)
            self.assertTrue(changed, source)
            self.assertEqual(flipped, expected)

    def test_flip_unsafe_items_are_not_counted_as_augmented(self):
        """Excluded items must not inflate the effective training count, since
        metadata['train_samples'] drives the global-step expectation."""
        from aicomp_grounding.query import count_effective_training_samples

        def item(query):
            return {
                "visible": "Raw/001/color/1.png",
                "infrared": "Raw/001/infrared/1.png",
                "depth": "Raw/001/depth/1.png",
                "query": query,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "width": 1920,
                "height": 1080,
            }

        data = {
            "a": item("The second flower from the left"),               # flip-safe
            "b": item("The second flower from the right"),              # flip-safe
            "c": item("The fourth letter from left to right"),          # glyph order
            "d": item("The shoe on the right foot of the person"),      # egocentric
            "e": item("The brown dog in the grass"),                    # no lateral term
        }
        self.assertEqual(count_effective_training_samples(data, augment_flip=True), 5 + 2)


if __name__ == "__main__":
    unittest.main()
