import json
import unittest

from aicomp_grounding.grounding.messages import (
    QWEN3VL_SYSTEM_PROMPT,
    QWEN3VL_USER_TEMPLATE,
    build_grounding_messages,
    build_training_messages,
)


def _grounding(visible, query):
    return build_grounding_messages(
        visible,
        query,
        system_prompt=QWEN3VL_SYSTEM_PROMPT,
        user_template=QWEN3VL_USER_TEMPLATE,
    )


class PromptTests(unittest.TestCase):
    def test_inference_messages_have_no_bbox_channel(self):
        messages = _grounding("rgb", "the red chair")
        serialized = json.dumps(messages)
        self.assertIn("the red chair", serialized)
        self.assertNotIn('"bbox"', serialized)
        self.assertNotIn("groundtruth", serialized.lower())

    def test_inference_messages_carry_exactly_one_image(self):
        messages = _grounding("rgb", "the red chair")
        images = [
            part
            for message in messages
            for part in message["content"]
            if part.get("type") == "image"
        ]
        self.assertEqual(len(images), 1)

    def test_training_messages_reuse_the_inference_prompt_prefix(self):
        prompt = _grounding("rgb", "the red chair")
        training = build_training_messages(
            "rgb",
            "the red chair",
            "(1,2),(3,4)",
            system_prompt=QWEN3VL_SYSTEM_PROMPT,
            user_template=QWEN3VL_USER_TEMPLATE,
        )
        self.assertEqual(training[:-1], prompt)
        self.assertEqual(training[-1]["role"], "assistant")

    def test_system_prompt_specifies_special_token_format(self):
        self.assertIn("box_start", QWEN3VL_SYSTEM_PROMPT)
        self.assertIn("box_end", QWEN3VL_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
