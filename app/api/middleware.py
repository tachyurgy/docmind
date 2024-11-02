import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status code, and duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
            client=request.client.host if request.client else "unknown",
        )

        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Adds an X-Response-Time header with the request duration in milliseconds."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return response
