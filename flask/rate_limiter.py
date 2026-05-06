from __future__ import annotations

from collections import defaultdict, deque
from math import ceil
from threading import Lock
from time import time


class FixedWindowRateLimiter:
    """Thread-safe fixed-window limiter for one key namespace."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int = 60,
        cleanup_interval_seconds: int = 300,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds

        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._last_cleanup = time()
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """
        Consume one request for `key` if allowed.
        Returns: (allowed, retry_after_seconds).
        """
        now = time()
        with self._lock:
            self._cleanup_if_needed(now)
            bucket = self._requests[key]
            self._prune_bucket(bucket, now)

            if len(bucket) >= self.max_requests:
                oldest = bucket[0]
                retry_after = max(1, ceil(self.window_seconds - (now - oldest)))
                return False, retry_after

            bucket.append(now)
            return True, 0

    def _prune_bucket(self, bucket: deque[float], now: float) -> None:
        threshold = now - self.window_seconds
        while bucket and bucket[0] <= threshold:
            bucket.popleft()

    def _cleanup_if_needed(self, now: float) -> None:
        if now - self._last_cleanup < self.cleanup_interval_seconds:
            return

        empty_keys: list[str] = []
        for key, bucket in self._requests.items():
            self._prune_bucket(bucket, now)
            if not bucket:
                empty_keys.append(key)

        for key in empty_keys:
            self._requests.pop(key, None)

        self._last_cleanup = now
