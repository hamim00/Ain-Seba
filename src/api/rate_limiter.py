"""
AinSeba - Rate Limiter
Simple in-memory rate limiter for API endpoints.
Uses a sliding window counter per IP address.
"""

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimitEntry:
    """Tracks request timestamps for a single client."""
    timestamps: list[float] = field(default_factory=list)


class RateLimiter:
    """
    In-memory sliding window rate limiter.

    Tracks requests per client key (IP address) and enforces
    a maximum number of requests within a time window.
    """

    def __init__(
        self,
        max_requests: int = 20,
        window_seconds: int = 60,
    ):
        """
        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Window size in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)

    def is_allowed(self, client_key: str) -> bool:
        """
        Check if a request is allowed for the given client.

        Args:
            client_key: Client identifier (typically IP address).

        Returns:
            True if request is allowed, False if rate limited.
        """
        now = time.time()
        entry = self._clients[client_key]

        # Remove timestamps outside the window
        cutoff = now - self.window_seconds
        entry.timestamps = [t for t in entry.timestamps if t > cutoff]

        # Check limit
        if len(entry.timestamps) >= self.max_requests:
            logger.warning(
                f"Rate limit exceeded for {client_key}: "
                f"{len(entry.timestamps)}/{self.max_requests} in {self.window_seconds}s"
            )
            return False

        # Record this request
        entry.timestamps.append(now)
        return True

    def remaining(self, client_key: str) -> int:
        """Get remaining requests for a client."""
        now = time.time()
        entry = self._clients[client_key]
        cutoff = now - self.window_seconds
        active = [t for t in entry.timestamps if t > cutoff]
        return max(0, self.max_requests - len(active))

    def reset(self, client_key: str = None):
        """Reset rate limit for a client or all clients."""
        if client_key:
            self._clients.pop(client_key, None)
        else:
            self._clients.clear()
