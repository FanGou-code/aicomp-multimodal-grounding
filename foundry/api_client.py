"""OpenAI-protocol multimodal client with bounded retry behavior.

Targets the annotation provider (Zhipu / GLM-4.6V); also compatible with any
OpenAI-protocol endpoint. """

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class APIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class APIKeySuspended(APIError):
    """Raised when a key is suspended mid-request for repeated transport failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=None)


@dataclass(frozen=True)
class APIResponse:
    content: str
    record: dict


class SlidingWindowRateLimiter:
    """Share conservative RPM and TPM budgets across annotation workers."""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        tokens_per_minute: int,
        estimated_tokens_per_request: int,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        limits = (
            requests_per_minute,
            tokens_per_minute,
            estimated_tokens_per_request,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in limits
        ):
            raise ValueError("Rate limits and estimated request tokens must be positive integers")
        if estimated_tokens_per_request > tokens_per_minute:
            raise ValueError("Estimated request tokens cannot exceed the per-minute token limit")
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.estimated_tokens_per_request = estimated_tokens_per_request
        self.clock = clock
        self.sleeper = sleeper
        self._lock = threading.Lock()
        self._next_id = 0
        self._events: list[list[float | int]] = []

    def acquire(self) -> int:
        while True:
            with self._lock:
                now = self.clock()
                self._events = [
                    event for event in self._events if now - float(event[0]) < 60.0
                ]
                used_tokens = sum(int(event[2]) for event in self._events)
                if (
                    len(self._events) < self.requests_per_minute
                    and used_tokens + self.estimated_tokens_per_request <= self.tokens_per_minute
                ):
                    self._next_id += 1
                    reservation_id = self._next_id
                    self._events.append(
                        [now, reservation_id, self.estimated_tokens_per_request]
                    )
                    return reservation_id
                delay = max(0.01, 60.0 - (now - float(self._events[0][0])))
            self.sleeper(delay)

    def record_usage(self, reservation_id: int, total_tokens: int) -> None:
        if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens <= 0:
            return
        with self._lock:
            for event in self._events:
                if int(event[1]) == reservation_id:
                    event[2] = total_tokens
                    return


def _usage(payload: object) -> dict[str, int]:
    source = payload if isinstance(payload, dict) else {}
    result: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = source.get(field, 0)
        result[field] = value if isinstance(value, int) and not isinstance(value, bool) else 0
    return result


class OpenAIProtocolClient:
    def __init__(
        self,
        *,
        api_key: str = "",
        model: str,
        base_url: str,
        timeout_seconds: float = 180.0,
        transport_attempts: int = 5,
        rate_limit_attempts: int = 8,
        opener: Callable = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        enable_thinking: bool | None = False,
        thinking_mode: str | None = None,
        json_mode: bool = True,
        key_pool: object | None = None,
    ) -> None:
        if key_pool is None and not api_key.strip():
            raise ValueError("API_KEY is empty")
        if not model or not base_url.startswith("https://"):
            raise ValueError("API model and HTTPS base URL are required")
        if timeout_seconds <= 0 or transport_attempts <= 0 or rate_limit_attempts <= 0:
            raise ValueError("Timeout, transport and rate-limit attempts must be positive")
        self.api_key = api_key
        self.key_pool = key_pool
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.transport_attempts = transport_attempts
        self.rate_limit_attempts = rate_limit_attempts
        self.opener = opener
        self.sleeper = sleeper
        self.rate_limiter = rate_limiter
        self.enable_thinking = enable_thinking
        self.thinking_mode = thinking_mode
        self.json_mode = json_mode

    def complete(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> APIResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if self.thinking_mode is not None:
            payload["thinking"] = {"type": self.thinking_mode}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if self.key_pool is None:
            return self._send_with_retries(body, self.api_key)
        while True:
            index, key = self.key_pool.current()

            def on_transport_failure() -> None:
                if self.key_pool.note_transport_failure(index):
                    raise APIKeySuspended(
                        f"key#{index + 1} suspended after repeated transport failures"
                    )

            try:
                result = self._send_with_retries(body, key, on_transport_failure=on_transport_failure)
            except APIKeySuspended:
                # The pool suspended this key after repeated transport
                # failures; the cursor already moved past it.
                continue
            except APIError as exc:
                if exc.status in (401, 402, 403):
                    self.key_pool.retire(index, f"HTTP {exc.status}")
                    continue
                if exc.status == 429 and self.key_pool.note_rate_limited(index):
                    continue
                raise
            self.key_pool.note_success(index)
            return result

    def _send_with_retries(
        self,
        body: bytes,
        api_key: str,
        on_transport_failure: Callable[[], None] | None = None,
    ) -> APIResponse:
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        transport_failures = 0
        rate_limit_retries = 0
        while transport_failures < self.transport_attempts:
            reservation_id = self.rate_limiter.acquire() if self.rate_limiter else None
            delay = 1.0
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    response_body = response.read()
                    headers = response.headers
                parsed = json.loads(response_body.decode("utf-8"))
                result = self._parse_response(parsed, headers)
                if self.rate_limiter and reservation_id is not None:
                    self.rate_limiter.record_usage(
                        reservation_id, result.record["usage"]["total_tokens"]
                    )
                return result
            except HTTPError as exc:
                detail = exc.read(2048).decode("utf-8", errors="replace")
                if exc.code == 429:
                    # A rate limit is temporary (provider docs: back off and
                    # retry), so it never consumes a transport attempt and
                    # never retires the key on its own.
                    rate_limit_retries += 1
                    if rate_limit_retries >= self.rate_limit_attempts:
                        raise APIError(
                            f"API HTTP 429 persisted after {rate_limit_retries} backoffs: {detail}",
                            status=429,
                        ) from exc
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        parsed_delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        parsed_delay = 0.0
                    delay = parsed_delay if parsed_delay > 0.0 else 2 ** (rate_limit_retries - 1)
                elif 500 <= exc.code < 600:
                    transport_failures += 1
                    if transport_failures == self.transport_attempts:
                        raise APIError(
                            f"API HTTP {exc.code}: {detail}", status=exc.code
                        ) from exc
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        parsed_delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        parsed_delay = 0.0
                    delay = parsed_delay if parsed_delay > 0.0 else 2 ** (transport_failures - 1)
                else:
                    raise APIError(
                        f"API HTTP {exc.code}: {detail}", status=exc.code
                    ) from exc
            except (TimeoutError, URLError) as exc:
                transport_failures += 1
                if on_transport_failure is not None:
                    # May raise APIKeySuspended to abort further attempts on
                    # this key after repeated hangs.
                    on_transport_failure()
                if transport_failures == self.transport_attempts:
                    raise APIError(f"API request failed: {exc}") from exc
                delay = 2 ** (transport_failures - 1)
            except APIError as exc:
                # Deliberately retried: transient empty-content responses from
                # the annotation model are recovered by re-asking (2026-08-23 fix).
                transport_failures += 1
                if transport_failures == self.transport_attempts:
                    raise
                delay = 2 ** (transport_failures - 1)
            self.sleeper(min(delay, 30.0) + random.random() * 0.25)
        raise AssertionError("Unreachable API retry state")

    def _parse_response(self, payload: object, headers: object) -> APIResponse:
        if not isinstance(payload, dict):
            raise APIError("API response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise APIError("API response must contain exactly one choice")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise APIError("API response was truncated by max_tokens")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise APIError("API response has no final content")
        trace_id = headers.get("x-siliconcloud-trace-id", "") if hasattr(headers, "get") else ""
        return APIResponse(
            content=content,
            record={
                "response_id": payload.get("id", ""),
                "trace_id": trace_id or "",
                "model": payload.get("model", self.model),
                "finish_reason": finish_reason or "",
                "usage": _usage(payload.get("usage")),
            },
        )
