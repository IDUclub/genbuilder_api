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
answer arrives as ``choices[0].delta.content``. Reasoning models (gpt-oss) put
their chain of thought in ``choices[0].delta.reasoning``, which is skipped.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"

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


class VLLMChatClient:
    """Thin async wrapper that streams assistant tokens from a vLLM server."""

    def __init__(
        self,
        base_url: str,
        *,
        default_model: str,
        timeout_seconds: float = 900.0,
        temperature: float = 0.3,
        reasoning_effort: str | None = "low",
    ) -> None:
        if not base_url:
            raise RuntimeError("LLM_API is not configured.")
        if not default_model:
            raise RuntimeError("a chat model must be configured (Chat_Model).")
        self._base_url = base_url.rstrip("/")
        self._chat_path = (
            "/chat/completions"
            if self._base_url.endswith("/v1")
            else "/v1/chat/completions"
        )
        self._default_model = default_model
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
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
            "messages": messages,
            "temperature": temperature,
        }
        effort = self._reasoning_effort if reasoning_effort is None else reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort
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
        Parses ``choices[0].message.content`` and returns it as a dict. Raises
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
        content = (choices[0].get("message") or {}).get("content") or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise VLLMChatError(resp.status_code, content) from exc
        if not isinstance(parsed, dict):
            raise VLLMChatError(resp.status_code, content)
        return parsed
