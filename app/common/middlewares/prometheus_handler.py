"""Observability middleware is defined here."""

import time

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.common.middlewares.middleware_utils import _normalize_path

from app.observability.metrics import Metrics


class ObservabilityMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: FastAPI, metrics: Metrics):
        """Obervability middleware class for http metrics with prometheus

        Args:
            app (FastAPI): FastAPI app instance
            metrics (Metrics): Metrics with http field connectable with prometheus
        """
        super().__init__(app)
        self._http_metrics = metrics.http

    async def dispatch(self, request: Request, call_next):
        method = request.method

        self._http_metrics.inflight_requests.add(1)

        start = time.monotonic()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            path = _normalize_path(request)

            duration = time.monotonic() - start

            self._http_metrics.requests_started.add(1, {"method": method, "path": path})
            self._http_metrics.requests_finished.add(
                1, {"method": method, "path": path, "status_code": status_code}
            )
            self._http_metrics.request_processing_duration.record(
                duration, {"method": method, "path": path}
            )
            self._http_metrics.inflight_requests.add(-1)