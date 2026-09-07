"""Offline tests for the API key pool and pooled client rotation."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

from foundry.pipeline.api import APIError, OpenAIProtocolClient
from foundry.pipeline.api import (
    APIKeyPool,
    APIKeyPoolExhausted,
    RATE_LIMIT_STRIKES_BEFORE_RETIRE,
    TRANSPORT_FAILURES_BEFORE_SUSPEND,
    comment_out_key,
    load_api_keys,
)


def _http_error(code: int) -> HTTPError:
    return HTTPError("https://example.invalid", code, "err", Message(), io.BytesIO(b"{}"))


class LoadKeyFileTests(unittest.TestCase):
    def test_parses_comments_blanks_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_keys.txt"
            path.write_text("# comment\n\nkey-one\n  key-two  \n# more\nkey-three\n")
            self.assertEqual(load_api_keys(path), ["key-one", "key-two", "key-three"])

    def test_env_fallback_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.txt"
            old = os.environ.get("ANNOTATION_API_KEYS")
            os.environ["ANNOTATION_API_KEYS"] = "env-a, env-b,,env-a"
            try:
                # loader preserves raw order; dedup happens in APIKeyPool
                self.assertEqual(load_api_keys(missing), ["env-a", "env-b", "env-a"])
            finally:
                if old is None:
                    os.environ.pop("ANNOTATION_API_KEYS", None)
                else:
                    os.environ["ANNOTATION_API_KEYS"] = old


class KeyPoolTests(unittest.TestCase):
    def test_rotation_cycles_through_alive_keys(self):
        pool = APIKeyPool(["k1", "k2", "k3"], notify=lambda m: None)
        order = [pool.current()[1] for _ in range(6)]
        self.assertEqual(order, ["k1", "k2", "k3", "k1", "k2", "k3"])

    def test_retired_keys_are_skipped_and_exhaustion_raises(self):
        messages = []
        pool = APIKeyPool(["k1", "k2"], notify=messages.append)
        pool.retire(0, "HTTP 402")
        self.assertEqual(pool.current(), (1, "k2"))
        self.assertEqual(pool.describe(), "1/2 alive")
        self.assertTrue(any("key#1 retired" in m for m in messages))
        pool.retire(1, "HTTP 402")
        with self.assertRaises(APIKeyPoolExhausted):
            pool.current()
        pool.retire(1, "HTTP 402")  # idempotent

    def test_transport_failures_suspend_and_success_resets(self):
        messages = []
        pool = APIKeyPool(["k1", "k2"], notify=messages.append)
        self.assertFalse(pool.note_transport_failure(0))
        pool.note_success(0)
        # success reset the counter, so one more failure is not enough
        self.assertFalse(pool.note_transport_failure(0))
        self.assertTrue(pool.note_transport_failure(0))
        self.assertEqual(pool.current(), (1, "k2"))
        self.assertTrue(any("suspended" in m for m in messages))

    def test_rate_limit_strikes_retire_after_threshold(self):
        pool = APIKeyPool(["k1", "k2"], notify=lambda m: None)
        for _ in range(RATE_LIMIT_STRIKES_BEFORE_RETIRE - 1):
            self.assertFalse(pool.note_rate_limited(0))
        self.assertTrue(pool.note_rate_limited(0))
        self.assertEqual(pool.current(), (1, "k2"))

    def test_rejects_empty_pool(self):
        with self.assertRaises(ValueError):
            APIKeyPool(["", "  "])
        with self.assertRaises(ValueError):
            APIKeyPool([])


class CommentOutKeyTests(unittest.TestCase):
    def test_comments_target_line_and_preserves_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_keys.txt"
            path.write_text("# header note\nkey-one\nkey-two\n# parked\n# key-three\n", encoding="utf-8")
            os.chmod(path, 0o600)
            self.assertTrue(comment_out_key(path, 1, "HTTP 402"))
            text = path.read_text(encoding="utf-8")
            self.assertIn("# key-two  # ", text)
            self.assertIn("auto-retired: HTTP 402", text)
            self.assertIn("# key-three", text)  # parked line untouched
            self.assertEqual(load_api_keys(path), ["key-one"])
            self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")

    def test_unknown_index_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_keys.txt"
            path.write_text("key-one\n")
            self.assertFalse(comment_out_key(path, 5, "HTTP 401"))


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


class PooledClientTests(unittest.TestCase):
    def _payload(self):
        return {
            "id": "r1",
            "model": "glm-4.6v",
            "choices": [
                {"message": {"content": '{"query":"test"}'}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def test_429_never_retires_and_backs_off_on_same_key(self):
        seen_keys = []

        def opener(request, timeout):
            key = request.headers["Authorization"].removeprefix("Bearer ")
            seen_keys.append(key)
            raise _http_error(429)

        pool = APIKeyPool(["k1", "k2"], notify=lambda m: None)
        client = OpenAIProtocolClient(
            key_pool=pool,
            model="glm-4.6v",
            base_url="https://api.example.invalid/v1",
            opener=opener,
            sleeper=lambda delay: None,
        )
        with self.assertRaises(APIError) as ctx:
            client.complete(
                messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=0.1
            )
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(len(seen_keys), 8)
        self.assertEqual(set(seen_keys), {"k1"})
        self.assertEqual(pool.alive(), 2)

    def test_sustained_429_retires_and_persists(self):
        def opener(request, timeout):
            raise _http_error(429)

        persisted = []
        pool = APIKeyPool(["k1"], notify=lambda m: None, persist_retire=lambda i, r: persisted.append((i, r)))
        client = OpenAIProtocolClient(
            key_pool=pool,
            model="glm-4.6v",
            base_url="https://api.example.invalid/v1",
            opener=opener,
            sleeper=lambda delay: None,
        )
        for _ in range(RATE_LIMIT_STRIKES_BEFORE_RETIRE):
            with self.assertRaises(APIError):
                client.complete(
                    messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=0.1
                )
        self.assertEqual(pool.alive(), 0)
        self.assertEqual(persisted, [(0, "sustained HTTP 429 x30")])
        with self.assertRaises(APIKeyPoolExhausted):
            client.complete(
                messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=0.1
            )

    def test_immediate_switch_on_auth_error(self):
        seen_keys = []

        def opener(request, timeout):
            key = request.headers["Authorization"].removeprefix("Bearer ")
            seen_keys.append(key)
            if key == "bad":
                raise _http_error(401)
            return _FakeResponse(self._payload())

        pool = APIKeyPool(["bad", "good"], notify=lambda m: None)
        client = OpenAIProtocolClient(
            key_pool=pool,
            model="glm-4.6v",
            base_url="https://api.example.invalid/v1",
            opener=opener,
            sleeper=lambda delay: None,
        )
        client.complete(
            messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=0.1
        )
        self.assertEqual(seen_keys, ["bad", "good"])

    def test_suspends_key_after_repeated_transport_failures(self):
        seen_keys = []

        def opener(request, timeout):
            key = request.headers["Authorization"].removeprefix("Bearer ")
            seen_keys.append(key)
            if key == "hang":
                raise TimeoutError("hung")
            return _FakeResponse(self._payload())

        pool = APIKeyPool(["hang", "good"], notify=lambda m: None)
        client = OpenAIProtocolClient(
            key_pool=pool,
            model="glm-4.6v",
            base_url="https://api.example.invalid/v1",
            opener=opener,
            sleeper=lambda delay: None,
            transport_attempts=2,
        )
        response = client.complete(
            messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=0.1
        )
        self.assertEqual(seen_keys, ["hang", "hang", "good"])
        self.assertEqual(pool.alive(), 1)
        self.assertEqual(response.record["response_id"], "r1")

    def test_server_error_keeps_same_key(self):
        seen_keys = []

        def opener(request, timeout):
            key = request.headers["Authorization"].removeprefix("Bearer ")
            seen_keys.append(key)
            raise _http_error(503)

        pool = APIKeyPool(["k1", "k2"], notify=lambda m: None)
        client = OpenAIProtocolClient(
            key_pool=pool,
            model="glm-4.6v",
            base_url="https://api.example.invalid/v1",
            opener=opener,
            sleeper=lambda delay: None,
            transport_attempts=2,
        )
        with self.assertRaises(APIError) as ctx:
            client.complete(
                messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=0.1
            )
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(seen_keys, ["k1", "k1"])
        self.assertEqual(pool.alive(), 2)


if __name__ == "__main__":
    unittest.main()

    def test_remote_disconnected_retries_and_succeeds(self):
        # Regression (full-run crash): the provider sometimes closes the
        # connection without a response (http.client.RemoteDisconnected) —
        # that is a transport failure and must retry, not crash the run.
        from http.client import RemoteDisconnected

        calls = []

        def opener(request, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise RemoteDisconnected("Remote end closed connection without response")
            return _FakeResponse(self._payload())

        pool = APIKeyPool(["k1"], notify=lambda m: None)
        client = OpenAIProtocolClient(
            key_pool=pool,
            model="glm-4.6v",
            base_url="https://api.example.invalid/v1",
            opener=opener,
            sleeper=lambda delay: None,
        )
        response = client.complete(
            messages=[{"role": "user", "content": "t"}], max_tokens=8, temperature=0.1
        )
        self.assertEqual(len(calls), 3)
        self.assertIn("content", response.content)
        self.assertEqual(pool.alive(), 1)


    def test_suspended_keys_get_one_second_wind(self):
        # A provider-wide blip suspends every key; the pool revives them once
        # instead of dying mid-run. Hard retirements never revive.
        pool = APIKeyPool(["k1", "k2"], notify=lambda m: None)
        pool.note_transport_failure(0)
        pool.note_transport_failure(0)
        pool.note_transport_failure(1)
        pool.note_transport_failure(1)
        self.assertEqual(pool.alive(), 0)
        self.assertEqual(pool.current(), (0, "k1"))  # revived
        self.assertEqual(pool.current(), (1, "k2"))
        # Second exhaustion after revival: no more wind, hard stop.
        pool.note_transport_failure(0)
        pool.note_transport_failure(0)
        pool.note_transport_failure(1)
        pool.note_transport_failure(1)
        with self.assertRaises(APIKeyPoolExhausted):
            pool.current()
        # Hard retirement (auth) is never revived even before exhaustion.
        pool2 = APIKeyPool(["a", "b"], notify=lambda m: None)
        pool2.retire(0, "HTTP 401")
        pool2.note_transport_failure(1)
        pool2.note_transport_failure(1)
        with self.assertRaises(APIKeyPoolExhausted):
            pool2.current()
