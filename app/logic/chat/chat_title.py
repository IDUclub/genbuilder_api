"""Short chat titles for the history list.

The first user message makes a poor title: "количество жителей 17" and
"количество жителей 20" are two indistinguishable rows in the sidebar. So the
title is written by the LLM in one short guided-decoding call — a noun phrase
naming what the chat is about, not a copy of the request.

The call is deliberately cheap and never load-bearing: it runs once, when the
chat is created, under its own timeout, and any failure (LLM down, junk output,
too slow) falls back to the trimmed user query — exactly what the title used to
be. A chat is never left uncreated because a title could not be generated.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

from app.infrastructure.vllm_chat_client import VLLMChatClient

# ChatStorage accepts long titles; the sidebar does not show them. Keep titles
# short enough to be read at a glance and truncate on a word boundary.
TITLE_MAX_CHARS = 60

# The title must never hold up chat creation — the LLM client's own timeout is
# minutes long, which is the wrong order of magnitude for one short phrase.
TITLE_TIMEOUT_SECONDS = 15.0

_TITLE_SYSTEM_PROMPT = (
    "Ты придумываешь короткий заголовок для чата о генерации городской "
    "застройки. Заголовок — именная группа из 3–5 слов на русском языке, "
    "по которой чат можно узнать в списке: что генерируем и главный параметр "
    "(например «Жильё на 5000 жителей» или «Застройка по своим кварталам»). "
    "Без кавычек, без точки в конце, без слов «чат», «запрос», «генерация "
    "застройки» саму по себе. Опирайся только на запрос пользователя, ничего "
    "не выдумывай: если в запросе нет деталей, опиши его как есть."
)

_TITLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
}

_WHITESPACE = re.compile(r"\s+")
_STRIP_CHARS = " \t\n\r\"'«»`*_.,:;—–-"


def _tidy(raw: object) -> str:
    """Collapse a model answer (or a raw query) into one clean short line."""
    text = _WHITESPACE.sub(" ", str(raw or "")).strip(_STRIP_CHARS)
    if len(text) <= TITLE_MAX_CHARS:
        return text
    cut = text[:TITLE_MAX_CHARS].rsplit(" ", 1)[0].strip(_STRIP_CHARS)
    return f"{cut or text[:TITLE_MAX_CHARS].strip()}…"


def fallback_title(user_query: str) -> str:
    """The pre-LLM behaviour: the user's own words, trimmed to one line."""
    return _tidy(user_query) or "Генерация застройки"


async def make_chat_title(
    llm_client: VLLMChatClient,
    *,
    user_query: str,
    model: str | None = None,
    fallback: str | None = None,
) -> str:
    """Ask the LLM for a short title; on any failure return the fallback.

    Never raises: a title is cosmetic, and the caller is about to create a chat
    the user is waiting on.
    """
    safe = fallback_title(fallback if fallback is not None else user_query)
    query = _WHITESPACE.sub(" ", str(user_query or "")).strip()
    if not query:
        return safe

    messages = [
        {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        raw = await asyncio.wait_for(
            llm_client.complete_json(
                messages, schema=_TITLE_SCHEMA, model=model, temperature=0.0
            ),
            timeout=TITLE_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - a title is cosmetic, never load-bearing
        logger.warning("chat title generation failed, using the query: {}", exc)
        return safe

    title = _tidy(raw.get("title"))
    if not title:
        logger.warning("chat title generation returned nothing, using the query")
        return safe
    return title


__all__ = ["TITLE_MAX_CHARS", "fallback_title", "make_chat_title"]
