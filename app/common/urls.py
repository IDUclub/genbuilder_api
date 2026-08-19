"""URL helpers for links that outlive the request that produced them."""
from __future__ import annotations


def durable_url(
    path: str,
    public_base_url: str | None,
    request_base_url: str | None = None,
) -> str:
    """Stable, never-expiring URL for an API path.

    Absolute when ``PUBLIC_BASE_URL`` is configured — which is what a link kept
    in chat history needs — otherwise derived from the incoming request, else a
    relative path.
    """
    if public_base_url:
        return f"{public_base_url.rstrip('/')}{path}"
    if request_base_url:
        return f"{request_base_url.rstrip('/')}{path}"
    return path
