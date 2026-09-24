import json
import unittest

from aicomp_grounding.messages import build_grounding_messages, build_training_messages, GROUNDING_SYSTEM_PROMPT


class PromptTests(unittest.TestCase):
    def test_inference_messages_have_no_bbox_channel(self):
        messages = build_grounding_messages("rgb", "ir", "depth", "the red chair")
        serialized = json.dumps(messages)
        self.assertIn("the red chair", serialized)
        self.assertNotIn('"bbox"', serialized)
        self.assertNotIn("groundtruth", serialized.lower())

    def test_training_messages_reuse_the_inference_prompt_prefix(self):
        prompt = build_grounding_messages("rgb", "ir", "depth", "the red chair")
        training = build_training_messages("rgb", "ir", "depth", "the red chair", "(1,2),(3,4)")
        self.assertEqual(training[:-1], prompt)
        self.assertEqual(training[-1]["role"], "assistant")

    def test_system_prompt_specifies_special_token_format(self):
        self.assertIn("box_start", GROUNDING_SYSTEM_PROMPT)
        self.assertIn("box_end", GROUNDING_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
