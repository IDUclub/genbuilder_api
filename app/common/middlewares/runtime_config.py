"""ASGI middleware that periodically syncs shared runtime configuration."""

from __future__ import annotations

from app.infrastructure.config_runtime import apply_overrides


class RuntimeConfigMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in {"http", "websocket"}:
            apply_overrides()
        await self.app(scope, receive, send)
