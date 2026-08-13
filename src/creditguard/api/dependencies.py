"""FastAPI dependencies: API-key authentication and per-key rate limiting.

Both are plain dependency callables rather than global middleware, so they
compose per-route -- `/health` stays unauthenticated simply by not
depending on `require_api_key`, instead of needing a path-exclusion list
maintained somewhere else.
"""

from __future__ import annotations

import time
from threading import Lock

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from creditguard.config import get_settings

# A real `SecurityBase` scheme (not a plain `Header(...)` parameter) is what
# makes FastAPI register an entry in the OpenAPI `securitySchemes` -- which
# is what makes Swagger UI (`/docs`) show the "Authorize" button at all.
# `auto_error=False` so a missing header falls through to `require_api_key`'s
# own check below, which raises a 401 with our own message/shape rather than
# FastAPI's default one.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: str | None = Security(_api_key_header)) -> str:
    """Authenticate via the `X-API-Key` header against `settings.api_key`.

    Raises:
        HTTPException(401): if the header is missing or doesn't match.
    """
    settings = get_settings()
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )
    return x_api_key


class RateLimiter:
    """Simple in-memory, per-key fixed-window-per-minute limiter.

    Not distributed and not persisted across process restarts -- adequate
    for a single-process educational deployment. A real multi-worker
    deployment would need a shared store (e.g. Redis) instead; out of scope
    here.
    """

    def __init__(self, requests_per_minute: int | None = None) -> None:
        self._configured_limit = requests_per_minute
        self._lock = Lock()
        # key -> (window_start_epoch_minute, count_in_that_window)
        self._windows: dict[str, tuple[int, int]] = {}

    def _limit(self) -> int:
        if self._configured_limit is not None:
            return self._configured_limit
        return get_settings().rate_limit_rpm

    def check(self, key: str) -> None:
        """Raise 429 if `key` has exceeded its per-minute request budget."""
        limit = self._limit()
        current_minute = int(time.time() // 60)
        with self._lock:
            window_start, count = self._windows.get(key, (current_minute, 0))
            if window_start != current_minute:
                window_start, count = current_minute, 0
            count += 1
            self._windows[key] = (window_start, count)
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({limit} requests/minute)",
            )

    def reset(self) -> None:
        """Clear all rate-limit state. Used between tests so unrelated test
        functions sharing the same API key don't interfere with each
        other's request counts.
        """
        with self._lock:
            self._windows.clear()


_rate_limiter = RateLimiter()


def enforce_rate_limit(api_key: str = Depends(require_api_key)) -> str:
    """Authenticate (`require_api_key`) then enforce that key's rate limit.
    The single dependency routes should use for "authenticated and within
    budget."
    """
    _rate_limiter.check(api_key)
    return api_key


def reset_rate_limiter() -> None:
    """Test-only hook: clear all in-memory rate-limit state."""
    _rate_limiter.reset()
