from fastapi import Request


def _normalize_path(request: Request) -> str:
    """Return stable route template path for metrics labels.

    Prefers FastAPI route template (e.g. /items/{item_id}) to avoid high-cardinality labels.
    Falls back to request.url.path if route is not available (e.g. 404).
    """
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path
    return request.url.path