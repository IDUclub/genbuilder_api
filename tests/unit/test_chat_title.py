"""Chat titles: short, distinguishable, and never load-bearing."""
import asyncio

import pytest

from app.infrastructure.vllm_chat_client import VLLMChatError
from app.logic.chat.chat_title import (
    TITLE_MAX_CHARS,
    fallback_title,
    make_chat_title,
)

QUERY = "количество жителей 5000"


class _FakeLLM:
    def __init__(self, title=None, exc=None, delay=0.0):
        self._title = title
        self._exc = exc
        self._delay = delay
        self.calls: list[dict] = []

    async def complete_json(self, messages, *, schema, model=None, temperature=0.0):
        self.calls.append({"messages": messages, "model": model})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc is not None:
            raise self._exc
        return {"title": self._title}


def _title(llm, **kwargs) -> str:
    return asyncio.run(make_chat_title(llm, user_query=QUERY, **kwargs))


def test_the_model_writes_the_title():
    assert _title(_FakeLLM("Генерация жилья на 5000 жителей")) == "Генерация жилья на 5000 жителей"


def test_the_user_query_is_the_only_grounding():
    llm = _FakeLLM("Генерация жилья на 5000 жителей")

    _title(llm)

    assert llm.calls[0]["messages"][-1] == {"role": "user", "content": QUERY}


def test_an_llm_failure_falls_back_to_the_query():
    assert _title(_FakeLLM(exc=VLLMChatError(503, "down"))) == QUERY


def test_an_unexpected_failure_falls_back_too():
    """The client does not wrap transport errors — a title must survive them."""
    assert _title(_FakeLLM(exc=ConnectionError("no route"))) == QUERY


def test_a_slow_model_does_not_hold_up_chat_creation(monkeypatch):
    monkeypatch.setattr("app.logic.chat.chat_title.TITLE_TIMEOUT_SECONDS", 0.01)

    assert _title(_FakeLLM("Генерация жилья на 5000 жителей", delay=0.5)) == QUERY


@pytest.mark.parametrize("answer", ["", "   ", None, "«»"])
def test_an_empty_answer_falls_back_to_the_query(answer):
    assert _title(_FakeLLM(answer)) == QUERY


def test_an_explicit_fallback_wins_over_the_query():
    assert _title(_FakeLLM(exc=VLLMChatError(503, "down")), fallback="Свой заголовок") == "Свой заголовок"


def test_quotes_and_trailing_punctuation_are_stripped():
    assert _title(_FakeLLM('"Генерация жилья на 5000 жителей".')) == "Генерация жилья на 5000 жителей"


def test_a_multiline_answer_becomes_one_line():
    assert _title(_FakeLLM("Генерация жилья\nна 5000\tжителей")) == "Генерация жилья на 5000 жителей"


def test_a_long_title_is_cut_on_a_word_boundary():
    title = _title(_FakeLLM("Жилая и многофункциональная застройка " * 5))

    assert len(title) <= TITLE_MAX_CHARS + 1  # the ellipsis
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")


def test_fallback_title_is_the_trimmed_query():
    assert fallback_title("  количество\n жителей 17  ") == "количество жителей 17"


def test_fallback_title_is_never_empty():
    assert fallback_title("   ") == "Генерация застройки"
