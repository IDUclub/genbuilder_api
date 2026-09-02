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


class VLLMChatError(RuntimeError):
    """Non-2xx response (or malformed stream) from vLLM ``/v1/chat/completions``."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"vllm /v1/chat/completions returned {status}: {body!r}")


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
        arrive. Raises ``VLLMChatError`` on a non-2xx status or unparseable
        stream.
        """
        payload = self._payload(
            messages,
            model=model,
            temperature=self._temperature if temperature is None else temperature,
            stream=True,
            reasoning_effort=reasoning_effort,
        )

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
        ``VLLMChatError`` on a non-2xx status or when the content is not a JSON
        object.
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
        resp = await self._client.post(self._chat_path, json=payload)
        if resp.status_code >= 400:
            raise VLLMChatError(resp.status_code, resp.text)
        choices = resp.json().get("choices") or []
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
