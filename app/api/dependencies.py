import time
from collections import defaultdict

import structlog
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import settings

logger = structlog.get_logger()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Validate the API key if authentication is enabled.

    When settings.api_key is empty, all requests are allowed through.
    When set, the request must include a matching X-API-Key header.
    """
    if not settings.auth_enabled:
        return None

    if api_key is None or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return api_key


class RateLimiter:
    """Simple in-memory token bucket rate limiter.

    Tracks request counts per client IP within a sliding window.
    Not suitable for multi-process deployments; use Redis-backed
    rate limiting in production at scale.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.requests_per_minute = requests_per_minute
        self._window_seconds = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_ip: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.monotonic()
        cutoff = now - self._window_seconds

        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > cutoff]

        if len(self._requests[client_ip]) >= self.requests_per_minute:
            return False

        self._requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(requests_per_minute=60)


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency that enforces rate limiting per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
