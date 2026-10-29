"""Async streaming client for a vLLM OpenAI-compatible ``/v1/chat/completions``.

Used by the conversational generation flow (see ``app.logic.chat``) to:
- extract structured generation parameters from the user's free text
  (``complete_json`` with a JSON schema, so the model can't hallucinate a field);
- stream a natural-language summary of the generation result token-by-token
  to the frontend via SSE (``stream_chat``).

Built on ``httpx`` (async streaming).

Request shape::

    POST /v1/chat/completions
    {"model": ..., "stream": true, "temperature": ...,
     "messages": [{"role", "content"}, ...]}

Streamed lines are SSE frames ``data: {...}`` ending with ``data: [DONE]``; the
answer arrives as ``choices[0].delta.content``. Thinking is disabled by default
through the chat template (including Qwen3); any reasoning deltas are skipped.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from loguru import logger

_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"
_EMPTY_CONTENT_ATTEMPTS = 2

# Pseudo-status for a failure that never reached the server (connect error,
# DNS failure, timeout) — there is no HTTP status to report in that case.
TRANSPORT_ERROR = 0


class VLLMChatError(RuntimeError):
    """Any failed call to vLLM ``/v1/chat/completions``.

    Covers a non-2xx response, a malformed stream, and — with
    ``status == TRANSPORT_ERROR`` — a transport failure (host down, DNS,
    timeout). Callers only ever have to catch this one type: an ``httpx`` error
    escaping the client would abort the SSE stream it is running inside.
    """

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        if status == TRANSPORT_ERROR:
            super().__init__(f"vllm /v1/chat/completions is unreachable: {body}")
        else:
            super().__init__(f"vllm /v1/chat/completions returned {status}: {body!r}")


def _transport_error(exc: httpx.HTTPError) -> VLLMChatError:
    """Wrap an ``httpx`` transport failure so callers catch one type only."""
    return VLLMChatError(TRANSPORT_ERROR, f"{type(exc).__name__}: {exc}")


class VLLMEmptyContentError(VLLMChatError):
    """A structured call has neither content nor a recoverable JSON answer."""


class VLLMChatClient:
    """Thin async wrapper that streams assistant tokens from a vLLM server."""

    def __init__(
        self,
        base_url: str,
        *,
        default_model: str,
        timeout_seconds: float = 900.0,
        temperature: float = 0.3,
        reasoning_effort: str | None = None,
        request_priority: int = -60,
        chat_priority: int = -100,
    ) -> None:
        if not base_url:
            raise RuntimeError("VLLM_BASE_URL is not configured.")
        if not default_model:
            raise RuntimeError("a chat model must be configured (CHAT_MODEL).")
        self._base_url = base_url.rstrip("/")
        self._chat_path = (
            "/chat/completions"
            if self._base_url.endswith("/v1")
            else "/v1/chat/completions"
        )
        self._default_model = default_model
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._request_priority = request_priority
        self._chat_priority = chat_priority
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout_seconds)

    async def __aenter__(self) -> "VLLMChatClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._client.aclose()

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None,
        temperature: float,
        stream: bool,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "stream": stream,
            "priority": self._chat_priority if stream else self._request_priority,
            "messages": messages,
            "temperature": temperature,
        }
        effort = self._reasoning_effort if reasoning_effort is None else reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort
        else:
            # Qwen defaults to thinking, but extraction and result summaries
            # need a direct final answer. Do not inherit GPT-OSS's effort setting.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream assistant content deltas for ``messages``.

        Yields the incremental ``choices[0].delta.content`` chunks as they
        arrive. Raises ``VLLMChatError`` on a non-2xx status, an unparseable
        stream, or a transport failure (the server could not be reached).
        """
        payload = self._payload(
            messages,
            model=model,
            temperature=self._temperature if temperature is None else temperature,
            stream=True,
            reasoning_effort=reasoning_effort,
        )

        try:
            async with self._client.stream("POST", self._chat_path, json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise VLLMChatError(resp.status_code, body.decode("utf-8", "replace"))
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith(_SSE_DATA_PREFIX):
                        continue
                    data = line[len(_SSE_DATA_PREFIX) :].strip()
                    if data == _SSE_DONE:
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise VLLMChatError(resp.status_code, data) from exc
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise _transport_error(exc) from exc

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Non-streaming completion with structured output (guided decoding).

        Sends ``stream: false`` and an OpenAI-style ``response_format`` of type
        ``json_schema`` so the model must return JSON conforming to ``schema``.
        Parses ``choices[0].message.content`` and returns it as a dict. A complete,
        schema-valid JSON answer in reasoning is recovered on a normally finished
        turn if content is empty; otherwise empty content is retried once. Raises
        ``VLLMChatError`` on a non-2xx status, a transport failure, or when the
        content is not a JSON object.
        """
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            stream=False,
            reasoning_effort=reasoning_effort,
        )
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
        for attempt in range(1, _EMPTY_CONTENT_ATTEMPTS):
            try:
                return await self._complete_json_once(payload, schema)
            except VLLMEmptyContentError as exc:
                logger.warning(
                    "vLLM structured call attempt {} returned no content, retrying: {}",
                    attempt,
                    exc,
                )
        return await self._complete_json_once(payload, schema)

    async def _complete_json_once(
        self, payload: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """One request/parse cycle; retry only empty structured answers."""
        try:
            resp = await self._client.post(self._chat_path, json=payload)
        except httpx.HTTPError as exc:
            raise _transport_error(exc) from exc
        if resp.status_code >= 400:
            raise VLLMChatError(resp.status_code, resp.text)
        try:
            body = resp.json()
        except ValueError as exc:
            raise VLLMChatError(resp.status_code, resp.text) from exc
        choices = body.get("choices") or []
        if not choices:
            raise VLLMChatError(resp.status_code, resp.text)
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if not content.strip():
            # Accept only a whole JSON answer, never an object embedded in a
            # reasoning plan, and never recover from a truncated completion.
            if choice.get("finish_reason") == "stop":
                for field in ("reasoning", "reasoning_content"):
                    reasoning = message.get(field)
                    if not isinstance(reasoning, str) or not reasoning.strip():
                        continue
                    try:
                        parsed = json.loads(reasoning)
                        if not isinstance(parsed, dict):
                            continue
                        validate_json_schema(parsed, schema)
                    except (ValueError, JsonSchemaValidationError):
                        continue
                    logger.warning(
                        "Recovered vLLM structured answer from {} after schema validation",
                        field,
                    )
                    return parsed
            raise VLLMEmptyContentError(
                resp.status_code,
                f"vLLM returned no message content: finish_reason={choice.get('finish_reason')!r}",
            )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise VLLMChatError(resp.status_code, content) from exc
        if not isinstance(parsed, dict):
            raise VLLMChatError(resp.status_code, content)
        return parsed
