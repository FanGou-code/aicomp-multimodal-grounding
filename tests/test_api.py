"""Offline tests for the single-key API client (foundry.pipeline.api)."""

from __future__ import annotations

import io
import json
import os
import unittest
from email.message import Message
from http.client import RemoteDisconnected
from unittest.mock import patch
from urllib.error import HTTPError

from aicomp_grounding.annotation.client import (
    ENV_API_KEY,
    MAX_API_CONCURRENCY,
    APIError,
    OpenAIProtocolClient,
    client_for,
    resolve_api_key,
    validate_concurrency,
)

BASE_URL = "https://api.example.invalid/v1"


def _http_error(code: int, headers: Message | None = None) -> HTTPError:
    return HTTPError(
        "https://example.invalid", code, "err", headers or Message(), io.BytesIO(b"{}")
    )


def _payload(content: str = '{"query": "test"}') -> dict:
    return {
        "id": "r1",
        "model": "glm-4.6v",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.headers = Message()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._body


def _client(opener, *, sleeper=None, **kwargs) -> OpenAIProtocolClient:
    return OpenAIProtocolClient(
        api_key="k1",
        model="glm-4.6v",
        base_url=BASE_URL,
        opener=opener,
        sleeper=sleeper or (lambda delay: None),
        **kwargs,
    )


def _call(client: OpenAIProtocolClient):
    return client.complete(
        messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=0.1
    )


class ResolveApiKeyTest(unittest.TestCase):
    def test_the_flag_wins_over_the_environment(self):
        with patch.dict(os.environ, {ENV_API_KEY: "from-env"}):
            self.assertEqual(resolve_api_key("from-flag"), "from-flag")

    def test_the_environment_supplies_it_when_the_flag_is_absent(self):
        with patch.dict(os.environ, {ENV_API_KEY: "  from-env  "}):
            self.assertEqual(resolve_api_key(None), "from-env")

    def test_no_key_anywhere_stops_the_run(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                resolve_api_key(None)
            with self.assertRaises(SystemExit):
                resolve_api_key("   ")


class ConcurrencyTest(unittest.TestCase):
    def test_the_injected_value_is_returned(self):
        self.assertEqual(validate_concurrency(1), 1)
        self.assertEqual(validate_concurrency(MAX_API_CONCURRENCY), MAX_API_CONCURRENCY)

    def test_out_of_range_or_non_integer_values_are_refused(self):
        for bad in (0, -1, MAX_API_CONCURRENCY + 1, 1.5, "8", True, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_concurrency(bad)


class ConstructionTest(unittest.TestCase):
    def test_an_empty_key_is_refused(self):
        with self.assertRaises(ValueError):
            OpenAIProtocolClient(api_key="  ", model="m", base_url=BASE_URL)

    def test_a_non_https_base_url_is_refused(self):
        with self.assertRaises(ValueError):
            OpenAIProtocolClient(api_key="k", model="m", base_url="http://example.invalid")


class RefusedKeyTest(unittest.TestCase):
    def test_a_refused_key_is_retried_then_stops_the_run(self):
        # No status gets a special verdict: every error is retried the same
        # bounded way and then surfaces, carrying the status and the body so the
        # operator can decide whether the key is dead.
        for code in (401, 402, 403):
            with self.subTest(code=code):
                calls = []

                def opener(request, timeout):
                    calls.append(1)
                    raise _http_error(code)

                with self.assertRaises(APIError) as ctx:
                    _call(_client(opener, transport_attempts=2))
                self.assertEqual(ctx.exception.status, code)
                self.assertEqual(len(calls), 2)
                self.assertIn(str(code), str(ctx.exception))


class RetryTest(unittest.TestCase):
    def test_server_errors_retry_the_same_key(self):
        calls = []

        def opener(request, timeout):
            calls.append(request.headers["Authorization"])
            raise _http_error(503)

        with self.assertRaises(APIError) as ctx:
            _call(_client(opener, transport_attempts=2))
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(calls, ["Bearer k1", "Bearer k1"])

    def test_remote_disconnected_retries_and_succeeds(self):
        # Regression (full-run crash): the provider sometimes closes the
        # connection without a response (http.client.RemoteDisconnected) —
        # that is a transport failure and must retry, not crash the run.
        calls = []

        def opener(request, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise RemoteDisconnected("Remote end closed connection without response")
            return _FakeResponse(_payload())

        response = _call(_client(opener))
        self.assertEqual(len(calls), 3)
        self.assertEqual(json.loads(response.content), {"query": "test"})

    def test_a_429_backs_off_and_keeps_the_same_key(self):
        calls = []
        delays = []

        def opener(request, timeout):
            calls.append(request.headers["Authorization"])
            raise _http_error(429)

        client = _client(opener, sleeper=delays.append)
        with self.assertRaises(APIError) as ctx:
            _call(client)
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(len(calls), client.rate_limit_attempts)
        self.assertEqual(set(calls), {"Bearer k1"})
        # One backoff between attempts; the last failure raises without sleeping.
        self.assertEqual(len(delays), client.rate_limit_attempts - 1)
        self.assertTrue(all(delay >= 1.0 for delay in delays))

    def test_a_429_then_success_returns_the_reply(self):
        calls = []

        def opener(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise _http_error(429, Message())
            return _FakeResponse(_payload())

        response = _call(_client(opener))
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(response.content), {"query": "test"})

    def test_a_retry_after_header_sets_the_wait(self):
        delays = []
        calls = []

        def opener(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                headers = Message()
                headers["Retry-After"] = "3"
                raise _http_error(429, headers)
            return _FakeResponse(_payload())

        _call(_client(opener, sleeper=delays.append))
        # The jitter is added on top; the header value is the floor.
        self.assertGreaterEqual(delays[0], 3.0)

    def test_empty_content_is_retried(self):
        calls = []

        def opener(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                return _FakeResponse(_payload(""))
            return _FakeResponse(_payload())

        response = _call(_client(opener))
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(response.content), {"query": "test"})


class ClientForTest(unittest.TestCase):
    stage = {"thinking_mode": "disabled", "response_format": None}

    def test_retry_switches_the_attempt_budgets(self):
        retrying = client_for(self.stage, api_key="k")
        once = client_for(self.stage, api_key="k", retry=False)
        self.assertGreater(retrying.transport_attempts, 1)
        self.assertGreater(retrying.rate_limit_attempts, 1)
        self.assertEqual((once.transport_attempts, once.rate_limit_attempts), (1, 1))


class ResponseShapeTest(unittest.TestCase):
    def test_a_truncated_reply_is_an_error(self):
        payload = _payload()
        payload["choices"][0]["finish_reason"] = "length"

        with self.assertRaises(APIError) as ctx:
            _call(_client(lambda request, timeout: _FakeResponse(payload)))
        self.assertIn("truncated", str(ctx.exception))

    def test_two_choices_are_an_error(self):
        payload = _payload()
        payload["choices"].append(dict(payload["choices"][0]))

        with self.assertRaises(APIError):
            _call(_client(lambda request, timeout: _FakeResponse(payload)))

    def test_the_record_carries_usage_and_reasoning(self):
        payload = _payload()
        payload["choices"][0]["message"]["reasoning_content"] = "thought"

        response = _call(_client(lambda request, timeout: _FakeResponse(payload)))
        self.assertEqual(response.record["usage"]["total_tokens"], 2)
        self.assertEqual(response.record["reasoning"], "thought")


class PayloadTest(unittest.TestCase):
    def _sent(self, **kwargs) -> dict:
        seen = {}

        def opener(request, timeout):
            seen.update(json.loads(request.data.decode("utf-8")))
            return _FakeResponse(_payload())

        _client(opener, **kwargs).complete(
            messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=None,
        )
        return seen

    def test_temperature_is_omitted_when_none(self):
        self.assertNotIn("temperature", self._sent())

    def test_the_thinking_switch_is_sent(self):
        self.assertEqual(self._sent(thinking_mode="disabled")["thinking"], {"type": "disabled"})
        self.assertNotIn("thinking", self._sent(thinking_mode=None))

    def test_json_mode_is_on_by_default(self):
        self.assertEqual(self._sent()["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", self._sent(json_mode=False))


if __name__ == "__main__":
    unittest.main()
