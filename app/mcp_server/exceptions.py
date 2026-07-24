"""Translate this app's ``HTTPException``s into MCP / JSON-RPC errors.

Without this, every tool would need its own ``try/except HTTPException``
boilerplate. With it, tools just call the orchestration layer and any
``http_exception(...)`` raised there (see
``app.exceptions.http_exception_wrapper``) surfaces to the LLM as a
structured ``McpError`` instead of an opaque 500.

JSON-RPC error code conventions used here:
- ``-32002`` Application-defined: auth token rejected (mirrors PzzCompareAPI's
  MCP server convention for this same condition).
- ``-32602`` Invalid params (client-side issue: 4xx).
- ``-32603`` Internal error (server-side issue: 5xx or unexpected exception).

See https://www.jsonrpc.org/specification#error_object.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Awaitable, Callable

from fastapi import HTTPException
from mcp import ErrorData, McpError


def _format_detail(detail: Any) -> str:
    """Best-effort short message from ``HTTPException.detail``.

    ``http_exception()`` in this app always builds ``{"msg", "input", "detail"}``,
    but we don't assume that shape here in case a plain HTTPException slips through.
    """
    if isinstance(detail, dict):
        msg = detail.get("msg")
        if isinstance(msg, str) and msg:
            extra = detail.get("detail")
            return f"{msg} ({extra})" if extra else msg
    if isinstance(detail, str):
        return detail
    return repr(detail)


def _http_exception_to_mcp(exc: HTTPException) -> McpError:
    if exc.status_code in (401, 403):
        return McpError(ErrorData(
            code=-32002,
            message=(
                "AUTH_TOKEN_EXPIRED: the caller's token was rejected "
                f"(HTTP {exc.status_code}). Ask for a fresh token and retry — do "
                "not retry with the same token."
            ),
        ))
    if 400 <= exc.status_code < 500:
        code = -32602
    else:
        code = -32603
    return McpError(ErrorData(code=code, message=f"{exc.status_code}: {_format_detail(exc.detail)}"))


def map_errors(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator: wrap an async MCP tool so this app's HTTP errors surface cleanly."""

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except HTTPException as exc:
            raise _http_exception_to_mcp(exc) from exc

    return wrapper
