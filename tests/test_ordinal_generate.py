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


class MockOrdinalReplyTests(unittest.TestCase):
    def _reply(self, messages: list[dict]) -> str:
        return get_adapter("mock").generate_messages(
            [messages], max_new_tokens=64, temperature=0.0
        )[0]

    def test_parse_prompt_gets_a_decodable_intent(self):
        reply = self._reply(ordinal_parse.build_parse_messages("the second car from the left"))
        self.assertIsNotNone(ordinal_parse.strict_json_object(reply))

    def test_enumerate_prompt_gets_a_decodable_list(self):
        reply = self._reply(ordinal_enumerate.build_enumerate_messages(object(), "car"))
        payload = ordinal_enumerate.decode_run(reply)
        self.assertEqual(payload["count"], len(payload["instances"]))

    def test_repeated_calls_agree_so_the_reconcile_gate_passes(self):
        messages = ordinal_enumerate.build_enumerate_messages(object(), "car")
        self.assertEqual(self._reply(messages), self._reply(messages))

    def test_mock_flow_produces_a_decision(self):
        messages = ordinal_enumerate.build_enumerate_messages(object(), "car")
        payload = ordinal_enumerate.decode_run(self._reply(messages))
        base = payload["instances"][0]["bbox"]
        decision = resolve_query(
            base,
            ordinal_parse.strict_json_object(self._reply(ordinal_parse.build_parse_messages("x"))),
            [payload["instances"], payload["instances"]],
            [payload["count"], payload["count"]],
        )
        self.assertEqual((decision.action, decision.bbox), ("keep", None))


if __name__ == "__main__":
    unittest.main()
