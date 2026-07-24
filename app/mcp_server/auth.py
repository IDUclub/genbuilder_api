"""Bearer-token extraction for MCP tools.

The MCP server is mounted in-process on the same FastAPI app that serves the
REST API (see ``app/main.py``), so a caller's ``Authorization`` header is
available on the underlying HTTP request. Tools that forward to
UrbanDB-backed endpoints (anything keyed by ``scenario_id``) need that token;
``/generate/by_territory``-style tools that operate on inline geometry don't.
"""
from __future__ import annotations

from fastmcp.server.dependencies import get_http_headers
from mcp import ErrorData, McpError

from app.utils.auth import authenticate_token
from fastapi import HTTPException


def extract_bearer_token() -> str | None:
    """Pull the raw bearer token from the incoming Authorization header, if any."""
    headers = get_http_headers(include={"authorization"})
    auth_header = headers.get("authorization", "")
    return auth_header[7:].strip() if auth_header.startswith("Bearer ") else None


async def require_verified_token() -> str:
    """Extract and verify the caller's token, raising a JSON-RPC error if absent/invalid.

    Mirrors the verification ``app.utils.auth.verify_token`` performs for REST
    routes, so an MCP tool call is held to the same auth bar as its REST
    counterpart.
    """
    token = extract_bearer_token()
    if not token:
        raise McpError(ErrorData(
            code=-32002,
            message=(
                "AUTH_TOKEN_EXPIRED: no bearer token was found in the Authorization "
                "header. This tool requires a valid Keycloak access token."
            ),
        ))
    try:
        user = await authenticate_token(token)
    except HTTPException as exc:
        if exc.status_code in (401, 403):
            raise McpError(ErrorData(
                code=-32002,
                message=(
                    "AUTH_TOKEN_EXPIRED: the caller's token was rejected "
                    f"(HTTP {exc.status_code}). Ask for a fresh token and retry."
                ),
            )) from exc
        raise McpError(ErrorData(code=-32603, message=f"authentication failed: {exc.detail}")) from exc
    return user.token
