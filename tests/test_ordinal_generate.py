"""Tests for the shared generation helper and the mock adapter's ordinal replies."""

from __future__ import annotations

import unittest

import torch

from aicomp_grounding.models import get_adapter
from aicomp_grounding.models.base import run_generation
from aicomp_grounding.ordinal import enumerate as ordinal_enumerate
from aicomp_grounding.ordinal import parse as ordinal_parse
from aicomp_grounding.ordinal.resolve import resolve_query


class _FakeProcessor:
    def __init__(self, decoded: list[str]):
        self.decoded = decoded
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def batch_decode(self, tokens, *, skip_special_tokens: bool):
        self.calls.append((tuple(tokens.shape), skip_special_tokens))
        return list(self.decoded)


class _FakeModel:
    device = "cpu"

    def __init__(self, generated):
        self.generated = generated
        self.kwargs: dict | None = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return self.generated


class RunGenerationTests(unittest.TestCase):
    def _call(self, *, temperature: float):
        inputs = {"input_ids": torch.zeros((2, 3), dtype=torch.long)}
        generated = torch.tensor([[1, 2, 3, 7], [4, 5, 6, 8]])
        model = _FakeModel(generated)
        processor = _FakeProcessor(["a", "b"])
        decoded = run_generation(
            model,
            processor,
            inputs,
            max_new_tokens=64,
            temperature=temperature,
            skip_special_tokens=True,
        )
        return decoded, model, processor

    def test_prompt_tokens_are_sliced_off_and_decode_flags_forwarded(self):
        decoded, _model, processor = self._call(temperature=0.0)
        self.assertEqual(decoded, ["a", "b"])
        self.assertEqual(processor.calls, [((2, 1), True)])

    def test_greedy_path_never_passes_a_temperature(self):
        _decoded, model, _processor = self._call(temperature=0.0)
        self.assertEqual(model.kwargs["max_new_tokens"], 64)
        self.assertFalse(model.kwargs["do_sample"])
        self.assertNotIn("temperature", model.kwargs)

    def test_sampled_path_passes_the_temperature(self):
        _decoded, model, _processor = self._call(temperature=0.7)
        self.assertTrue(model.kwargs["do_sample"])
        self.assertAlmostEqual(model.kwargs["temperature"], 0.7)


    def test_sampled_path_forwards_the_sampling_kwargs(self):
        inputs = {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
        model = _FakeModel(torch.tensor([[1, 2, 3, 7]]))
        run_generation(
            model,
            _FakeProcessor(["a"]),
            inputs,
            max_new_tokens=64,
            temperature=0.6,
            sampling_kwargs={"top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
        )
        self.assertEqual(model.kwargs["top_p"], 0.95)
        self.assertEqual(model.kwargs["top_k"], 20)
        self.assertEqual(model.kwargs["presence_penalty"], 0.0)

    def test_greedy_path_drops_the_sampling_kwargs(self):
        inputs = {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
        model = _FakeModel(torch.tensor([[1, 2, 3, 7]]))
        run_generation(
            model,
            _FakeProcessor(["a"]),
            inputs,
            max_new_tokens=64,
            temperature=0.0,
            sampling_kwargs={"top_p": 0.95},
        )
        self.assertNotIn("top_p", model.kwargs)


class MockOrdinalReplyTests(unittest.TestCase):
    def _reply(self, messages: list[dict], temperature: float = 0.0, **kwargs) -> str:
        return get_adapter("mock").generate_messages(
            [messages], max_new_tokens=64, temperature=temperature, **kwargs
        )[0]

    def test_parse_prompt_gets_a_decodable_intent(self):
        reply = self._reply(ordinal_parse.build_parse_messages("the second car from the left"))
        self.assertIsNotNone(ordinal_parse.strict_json_object(reply))

    def test_enumerate_prompt_gets_a_decodable_list(self):
        reply = self._reply(ordinal_enumerate.build_enumerate_messages(object(), "car"))
        payload = ordinal_enumerate.decode_run(reply)
        self.assertEqual(payload["count"], len(payload["instances"]))

    def test_thinking_wraps_the_list_and_splits_back_to_it(self):
        reply = self._reply(
            ordinal_enumerate.build_enumerate_messages(object(), "car"),
            0.6,
            template_kwargs={"enable_thinking": True},
        )
        thinking, answer = ordinal_parse.split_thinking(reply)
        self.assertIsNotNone(thinking)
        payload = ordinal_enumerate.decode_run(answer)
        self.assertEqual(payload["count"], len(payload["instances"]))
        self.assertIn(payload["count"], ordinal_parse.thinking_reported_counts(thinking))

    def test_mock_flow_produces_a_decision(self):
        messages = ordinal_enumerate.build_enumerate_messages(object(), "car")
        payload = ordinal_enumerate.decode_run(self._reply(messages))
        base = payload["instances"][0]["bbox"]
        decision = resolve_query(
            base,
            ordinal_parse.strict_json_object(self._reply(ordinal_parse.build_parse_messages("x"))),
            payload,
        )
        self.assertEqual((decision.action, decision.bbox), ("keep", None))


if __name__ == "__main__":
    unittest.main()
