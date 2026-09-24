"""OpenAI-protocol multimodal client with bounded retry behavior.

Targets the annotation provider (Zhipu / GLM-4.6V); also compatible with any
OpenAI-protocol endpoint. One key per process, injected by the operator: the
client carries it and never rotates, retires or persists credentials.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


#: Credentials come from the environment, never from the repository.
ENV_API_KEY = "ANNOTATION_API_KEY"


class APIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class APIResponse:
    content: str
    record: dict


def resolve_api_key(cli_value: str | None = None) -> str:
    """The key for this run: the CLI flag if given, else ``$ANNOTATION_API_KEY``."""
    key = (cli_value or "").strip() or os.environ.get(ENV_API_KEY, "").strip()
    if not key:
        raise SystemExit(
            f"No API key: pass --api-key or set ${ENV_API_KEY} for this run."
        )
    return key


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
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 180.0,
        transport_attempts: int = 5,
        rate_limit_attempts: int = 8,
        opener: Callable = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        enable_thinking: bool | None = False,
        thinking_mode: str | None = None,
        json_mode: bool = True,
    ) -> None:
        if not api_key.strip():
            raise ValueError("API key is empty")
        if not model or not base_url.startswith("https://"):
            raise ValueError("API model and HTTPS base URL are required")
        if timeout_seconds <= 0 or transport_attempts <= 0 or rate_limit_attempts <= 0:
            raise ValueError("Timeout, transport and rate-limit attempts must be positive")
        self.api_key = api_key
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.transport_attempts = transport_attempts
        self.rate_limit_attempts = rate_limit_attempts
        self.opener = opener
        self.sleeper = sleeper
        self.enable_thinking = enable_thinking
        self.thinking_mode = thinking_mode
        self.json_mode = json_mode

    def complete(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float | None,
        do_sample: bool | None = None,
    ) -> APIResponse:
        """One chat completion.

        ``temperature=None`` and ``do_sample=None`` omit the field so the
        provider applies its own default.  ``do_sample=False`` is greedy and
        ignores temperature.  Safe to call from several threads: nothing is
        shared between calls but the immutable configuration.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if do_sample is not None:
            payload["do_sample"] = do_sample
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if self.thinking_mode is not None:
            payload["thinking"] = {"type": self.thinking_mode}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        return self._send_with_retries(body)

    def _send_with_retries(self, body: bytes) -> APIResponse:
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        transport_failures = 0
        rate_limit_retries = 0
        while transport_failures < self.transport_attempts:
            delay = 1.0
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    response_body = response.read()
                    headers = response.headers
                parsed = json.loads(response_body.decode("utf-8"))
                return self._parse_response(parsed, headers)
            except HTTPError as exc:
                detail = exc.read(2048).decode("utf-8", errors="replace")
                if exc.code in (401, 402, 403):
                    raise APIError(
                        f"API HTTP {exc.code} (the injected key was refused): {detail}",
                        status=exc.code,
                    ) from exc
                if exc.code == 429:
                    # A rate limit is temporary (provider docs: back off and
                    # retry), so it never consumes a transport attempt.
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
            except (TimeoutError, URLError, ConnectionResetError, HTTPException) as exc:
                # ConnectionResetError covers http.client.RemoteDisconnected
                # (provider closes the connection without a response); other
                # HTTPException subtypes (BadStatusLine, IncompleteRead) are
                # the same class of mid-protocol transport breakage. Both are
                # transient and retry with backoff like any transport failure.
                transport_failures += 1
                if transport_failures == self.transport_attempts:
                    raise APIError(f"API request failed: {exc}") from exc
                delay = 2 ** (transport_failures - 1)
            except APIError:
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
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        trace_id = headers.get("x-siliconcloud-trace-id", "") if hasattr(headers, "get") else ""
        return APIResponse(
            content=content,
            record={
                "response_id": payload.get("id", ""),
                "trace_id": trace_id or "",
                "model": payload.get("model", self.model),
                "finish_reason": finish_reason or "",
                "reasoning": reasoning if isinstance(reasoning, str) else "",
                "usage": _usage(payload.get("usage")),
            },
        )
