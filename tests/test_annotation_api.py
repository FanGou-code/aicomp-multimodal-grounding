"""Offline tests for marked generation views and API request handling."""

from __future__ import annotations

import json
import unittest
from email.message import Message
from PIL import Image
from scripts import generate_queries

from aicomp_grounding.annotation_state import (
    MARK_COLOR,
    build_marked_annotation_view,
    jpeg_data_url,
)
from aicomp_grounding.api_client import (
    OpenAIProtocolClient,
    APIResponse,
    SlidingWindowRateLimiter,
)


class AnnotationViewTests(unittest.TestCase):
    def test_marked_view_is_single_high_resolution_copy_with_outward_red_box(self):
        image = Image.new("RGB", (320, 180), "white")
        for x in range(150, 170):
            for y in range(80, 100):
                image.putpixel((x, y), (10, 40, 220))
        before = image.tobytes()
        marked = build_marked_annotation_view(
            image, [150 / 320, 80 / 180, 170 / 320, 100 / 180]
        )

        self.assertEqual(image.tobytes(), before)
        self.assertEqual(marked.size, image.size)
        flattened = list(marked.get_flattened_data())
        self.assertTrue(any(pixel == MARK_COLOR for pixel in flattened))
        self.assertTrue(any(pixel[2] > pixel[0] for pixel in flattened))
        self.assertTrue(jpeg_data_url(marked).startswith("data:image/jpeg;base64,"))


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.headers = Message()
        self.headers["x-siliconcloud-trace-id"] = "trace-123"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._body


class SiliconFlowClientTests(unittest.TestCase):
    def test_request_disables_thinking_uses_json_mode_and_records_billing_usage(self):
        captured = {}

        def opener(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["authorization"] = request.headers["Authorization"]
            captured["timeout"] = timeout
            return _FakeResponse(
                {
                    "id": "response-1",
                    "model": "glm-4.6v",
                    "choices": [
                        {
                            "message": {"content": '{"query":"test"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 12,
                        "total_tokens": 132,
                    },
                }
            )

        client = OpenAIProtocolClient(
            api_key="secret-test-key",
            model="glm-4.6v",
            base_url="https://api.siliconflow.cn/v1",
            opener=opener,
            thinking_mode="disabled",
        )
        response = client.complete(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=256,
            temperature=0.2,
        )

        self.assertFalse(captured["payload"]["enable_thinking"])
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["payload"]["max_tokens"], 256)
        self.assertEqual(captured["authorization"], "Bearer secret-test-key")
        self.assertEqual(response.record["trace_id"], "trace-123")
        self.assertEqual(response.record["usage"]["total_tokens"], 132)
        self.assertNotIn("secret-test-key", json.dumps(response.record))

    def test_shared_rate_limiter_honors_measured_token_usage(self):
        now = [0.0]
        sleeps = []

        def sleeper(delay):
            sleeps.append(delay)
            now[0] += delay

        limiter = SlidingWindowRateLimiter(
            requests_per_minute=10,
            tokens_per_minute=100,
            estimated_tokens_per_request=40,
            clock=lambda: now[0],
            sleeper=sleeper,
        )
        reservation = limiter.acquire()
        limiter.record_usage(reservation, 70)
        limiter.acquire()
        self.assertEqual(sleeps, [60.0])


class FrameGenerationTests(unittest.TestCase):
    class FakeClient:
        def __init__(self, responses, model):
            self.responses = list(responses)
            self.model = model
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            record = {
                "response_id": "response",
                "trace_id": "trace",
                "model": self.model,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            return APIResponse(self.responses.pop(0), record)

    @staticmethod
    def _marked():
        image = Image.new("RGB", (320, 180), "gray")
        return build_marked_annotation_view(image, [0.1, 0.1, 0.4, 0.6])

    @staticmethod
    def _style_fields(style="attribute_action"):
        family = {
            "attribute_action": "ATTRIBUTE_ACTION",
            "scene_location": "SCENE_REGION",
            "ordinal": "FROM_LEFT_TO_RIGHT",
        }[style]
        return {
            "annotation_style": style,
            "annotation_style_family": family,
            "annotation_min_words": 7,
            "annotation_max_words": 13,
            "annotation_fallback_style": "scene_location",
        }

    def test_single_model_generates_query_with_one_image(self):
        marked = self._marked()
        client = self.FakeClient(
            [json.dumps({
                "query": "The pedestrian wearing a yellow waterproof jacket",
                "alternate_query": None,
                "uncertain": False,
            })],
            "glm-4.6v",
        )
        result = generate_queries._annotate_frame(
            client,
            marked,
            previous=None,
            retry_failed=False,
            style_fields=self._style_fields(),
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["query"], "The pedestrian wearing a yellow waterproof jacket")
        self.assertFalse(result["uncertain"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["error"], "")
        self.assertEqual(len(result["api_calls"]), 1)
        content = client.calls[0]["messages"][1]["content"]
        self.assertEqual(
            sum(part["type"] == "image_url" for part in content),
            1,
        )
        text = " ".join(part["text"] for part in content if part["type"] == "text")
        self.assertIn("red rectangle", text)

    def test_invalid_query_retries_with_error_feedback(self):
        marked = self._marked()
        client = self.FakeClient(
            [
                json.dumps({
                    "query": "thing",
                    "alternate_query": None,
                    "uncertain": False,
                }),
                json.dumps({
                    "query": "The small brown monkey beside the rocks",
                    "alternate_query": None,
                    "uncertain": False,
                }),
            ],
            "glm-4.6v",
        )
        result = generate_queries._annotate_frame(
            client,
            marked,
            previous=None,
            retry_failed=False,
            style_fields=self._style_fields(),
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["attempts"], 2)
        self.assertIn("monkey", result["query"])
        self.assertEqual(len(client.calls), 2)
        second_prompt = client.calls[1]["messages"][1]["content"][-1]["text"]
        self.assertIn("failed deterministic QC", second_prompt)

    def test_all_attempts_exhausted_returns_failed(self):
        marked = self._marked()
        client = self.FakeClient(
            [
                json.dumps({"query": "entity", "alternate_query": None, "uncertain": False}),
                json.dumps({"query": "thing", "alternate_query": None, "uncertain": False}),
                json.dumps({"query": "item", "alternate_query": None, "uncertain": False}),
            ],
            "glm-4.6v",
        )
        result = generate_queries._annotate_frame(
            client,
            marked,
            previous=None,
            retry_failed=False,
            style_fields=self._style_fields(),
        )

        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["query"])
        self.assertEqual(result["attempts"], 3)
        self.assertTrue(result["error"])

    def test_progress_replaces_failed_status_on_retry(self):
        progress = generate_queries.AnnotationProgress(total=3)
        progress.sync({
            "001": {
                "frames": {
                    "001_1": {"status": "completed"},
                    "001_2": {"status": "failed"},
                }
            }
        })
        self.assertEqual(progress.update("001_2", "completed"), (2, 2, 0))
        self.assertEqual(progress.update("001_3", "failed"), (3, 2, 1))

    def test_style_gate_retries_bare_label_and_accepts_annotated_query(self):
        marked = self._marked()
        client = self.FakeClient(
            [
                # first attempt: 1-word label → style gate rejects
                json.dumps({"query": "cone", "alternate_query": None, "uncertain": False}),
                # second attempt: passes style gate
                json.dumps({"query": "The third cone from left to right in the row", "alternate_query": None, "uncertain": False}),
            ],
            "glm-4.6v",
        )
        result = generate_queries._annotate_frame(
            client,
            marked,
            previous=None,
            retry_failed=False,
            style_fields=self._style_fields(),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["attempts"], 2)
        self.assertIn("cone", result["query"])
        self.assertEqual(len(client.calls), 2)

    def test_style_fields_inject_group_specific_prompt(self):
        marked = self._marked()
        client = self.FakeClient(
            [
                json.dumps({
                    "query": "The third cone from left to right",
                    "alternate_query": None,
                    "uncertain": False,
                    "style": "ordinal",
                })
            ],
            "glm-4.6v",
        )
        result = generate_queries._annotate_frame(
            client,
            marked,
            previous=None,
            retry_failed=False,
            style_fields={
                "annotation_style": "ordinal",
                "annotation_style_family": "FROM_LEFT_TO_RIGHT",
                "annotation_min_words": 8,
                "annotation_max_words": 12,
                "annotation_fallback_style": "scene_location",
            },
        )
        self.assertEqual(result["status"], "completed")
        self.assertIn("FROM_LEFT_TO_RIGHT", client.calls[0]["messages"][1]["content"][-1]["text"])

    def test_verify_rejects_low_iou_and_retries(self):
        plain = Image.new("RGB", (320, 180), "gray")
        marked = build_marked_annotation_view(plain, [0.1, 0.1, 0.4, 0.6])
        gt_bbox = [0.1, 0.1, 0.4, 0.6]

        # Both generation responses pass the style gate (>=5 words, with
        # spatial cues), so both attempts reach the verification step.
        client = self.FakeClient(
            [
                # attempt 1 gen: valid, verify will fail (box far from GT)
                json.dumps({"query": "The dark gray patch beside the wall", "alternate_query": None, "uncertain": False}),
                # attempt 1 verify: returns a box far from GT → IoU below threshold
                '{"bbox":[0.8,0.8,0.95,0.95]}',
                # attempt 2 gen: valid
                json.dumps({"query": "The gray rectangle in the top-left corner", "alternate_query": None, "uncertain": False}),
                # attempt 2 verify: box near GT → passes
                '{"bbox":[0.12,0.12,0.38,0.58]}',
            ],
            "glm-4.6v",
        )
        result = generate_queries._annotate_frame(
            client,
            marked,
            previous=None,
            retry_failed=False,
            plain_rgb=plain,
            gt_bbox=gt_bbox,
            verify=True,
            style_fields=self._style_fields(style="scene_location"),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["attempts"], 2)
        # 2 gen calls + 2 verify calls = 4 total
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(result["api_calls"]), 4)

    def test_verify_passes_with_valid_box_on_first_try(self):
        plain = Image.new("RGB", (320, 180), "gray")
        marked = build_marked_annotation_view(plain, [0.1, 0.1, 0.4, 0.6])
        gt_bbox = [0.1, 0.1, 0.4, 0.6]

        client = self.FakeClient(
            [
                json.dumps({"query": "The gray rectangle in the top-left corner", "alternate_query": None, "uncertain": False}),
                '{"bbox":[0.11,0.11,0.39,0.59]}',  # verify → IoU ~ 0.86
            ],
            "glm-4.6v",
        )
        result = generate_queries._annotate_frame(
            client,
            marked,
            previous=None,
            retry_failed=False,
            plain_rgb=plain,
            gt_bbox=gt_bbox,
            verify=True,
            style_fields=self._style_fields(style="scene_location"),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["attempts"], 1)
        # 1 gen + 1 verify = 2 api_calls
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(result["api_calls"]), 2)
        verify_content = client.calls[1]["messages"][1]["content"]
        self.assertEqual(verify_content[0]["type"], "image_url")
        verify_text = " ".join(
            part["text"] for part in verify_content if part["type"] == "text"
        )
        self.assertIn("Referring query:", verify_text)


if __name__ == "__main__":
    unittest.main()
