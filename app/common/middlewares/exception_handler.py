"""Exception handling middleware is defined here."""

import traceback

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.common.middlewares.middleware_utils import _normalize_path
from app.observability.metrics import Metrics


class ExceptionHandlerMiddleware(
    BaseHTTPMiddleware
):  # pylint: disable=too-few-public-methods
    """Handle exceptions, so they become http response code 500 - Internal Server Error if not handled as HTTPException
    previously.
    Attributes:
           app (FastAPI): The FastAPI application instance.
    """

    def __init__(self, app: FastAPI, metrics: Metrics):
        """
        Universal exception handler middleware init function.
        Args:
            app (FastAPI): The FastAPI application instance.
        """

        super().__init__(app)
        self.metrics = metrics

    @staticmethod
    async def prepare_request_info(request: Request) -> dict:
        """
        Function prepares request input data
        Args:
            request (Request): Request instance.
        Returns:
            dict: Request input data.
        """

        request_info = {
            "method": request.method,
            "url": str(request.url),
            "path_params": dict(request.path_params),
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
        }

        try:
            request_info["body"] = await request.json()
            return request_info
        except:
            try:
                request_info["body"] = str(await request.body())
                return request_info
            except:
                request_info["body"] = "Could not read request body"
                return request_info

    async def dispatch(self, request: Request, call_next):
        """
        Dispatch function for sending errors to user from API
        Args:
            request (Request): The incoming request object.
            call_next: function to extract.
        """

        try:
            return await call_next(request)

        except Exception as e:
            request_info = await self.prepare_request_info(request)
            self.metrics.http.errors.add(
                1,
                {
                    "method": request.method,
                    "path": _normalize_path(request),
                    "error_type": type(e).__name__,
                    "status_code": 500,
                },
            )
            return JSONResponse(
                status_code=500,
                content={
                    "message": "Internal server error",
                    "error_type": e.__class__.__name__,
                    "request": request_info,
                    "detail": str(e),
                    "traceback": traceback.format_exc().splitlines(),
                },
            )