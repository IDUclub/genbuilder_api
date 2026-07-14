"""Async client for the external chat-history service (ChatStorage).

Persists the conversational generation dialogue so the frontend can keep a
multi-turn history and the LLM can be grounded on previous turns. Mirrors the
ChatStorage contract used by IDUclub/PzzCompareAPI.

Authentication is **machine-to-machine**: every request carries the service's
own Keycloak token (client-credentials, obtained via ``idu-service-auth``) plus
an ``X-User-Id`` header naming the end user. ChatStorage recognizes the service
account (its ``preferred_username`` is ``service-account-*``) and stores/reads
history under the supplied ``X-User-Id`` (the user's Keycloak ``sub``).

Endpoints::

    POST /api/v1/chat_history/create_chat        -> ChatSummary  {chat_id, title, ...}
    POST /api/v1/chat_history/{chat_id}/message  -> Message      {message_id, ...}
    GET  /api/v1/chat_history/{chat_id}          -> Chat         {messages: [...], ...}

Persistence is best-effort in the caller: a ``ChatStorageError`` is surfaced as
a non-fatal ``warning`` SSE event and never aborts the token stream.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from idu_service_auth import KeycloakTokenClient


class ChatStorageError(RuntimeError):
    """Non-2xx response (or transport error) from the chat-history service."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"chat_storage returned {status}: {body!r}")


class ChatStorageClient:
    """Thin async wrapper over the chat-history REST service."""

    def __init__(
        self,
        base_url: str,
        token_client: "KeycloakTokenClient",
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url:
            raise RuntimeError("ChatStorage_API is not configured.")
        if token_client is None:
            raise RuntimeError("A Keycloak service token client is required.")
        self._base_url = base_url.rstrip("/")
        self._token_client = token_client
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout_seconds)

    async def __aenter__(self) -> "ChatStorageClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._client.aclose()

    async def _auth_headers(self, user_id: str) -> dict[str, str]:
        """Service bearer token + the end-user id for ChatStorage."""
        # Import lazily so the module stays importable without the auth library.
        from idu_service_auth import KeycloakAuthError

        try:
            headers = dict(await self._token_client.get_authorization_headers())
        except KeycloakAuthError as exc:  # Keycloak unavailable / bad credentials
            raise ChatStorageError(0, f"service token unavailable: {exc}") from exc
        headers["X-User-Id"] = str(user_id)
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        user_id: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = await self._auth_headers(user_id)
        try:
            resp = await self._client.request(
                method, path, headers=headers, json=json_body
            )
        except httpx.HTTPError as exc:  # network / timeout
            raise ChatStorageError(0, str(exc)) from exc
        if resp.status_code >= 400:
            raise ChatStorageError(resp.status_code, resp.text)
        if not resp.content:
            return {}
        return resp.json()

    async def create_chat(
        self,
        user_id: str,
        *,
        title: str | None = None,
        scenario_id: str | int | None = None,
        project_id: str | int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "title": title,
            "scenario_id": scenario_id,
            "project_id": project_id,
            "metadata": metadata,
        }
        return await self._request(
            "POST", "/api/v1/chat_history/create_chat", user_id, json_body=body
        )

    async def add_message(
        self,
        user_id: str,
        chat_id: str,
        *,
        role: str,
        content: str | None = None,
        parts: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"role": role, "metadata": metadata}
        if parts is not None:
            body["parts"] = parts
        else:
            body["content"] = content
        return await self._request(
            "POST",
            f"/api/v1/chat_history/{chat_id}/message",
            user_id,
            json_body=body,
        )

    async def get_chat(self, user_id: str, chat_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/api/v1/chat_history/{chat_id}", user_id
        )
