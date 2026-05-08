from __future__ import annotations

from collections import defaultdict, deque
from math import ceil
from threading import Lock
from time import time


class FixedWindowRateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: int = 60,
        cleanup_interval_seconds: int = 300,
        backend: str = "memory",
        redis_url: str | None = None,
        key_prefix: str = "plank-agent:ratelimit",
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        backend = (backend or "memory").strip().lower()
        if backend == "redis":
            self._backend = _RedisFixedWindowBackend(
                max_requests=max_requests,
                window_seconds=window_seconds,
                redis_url=redis_url,
                key_prefix=key_prefix,
            )
        else:
            self._backend = _InMemoryFixedWindowBackend(
                max_requests=max_requests,
                window_seconds=window_seconds,
                cleanup_interval_seconds=cleanup_interval_seconds,
            )

    def allow(self, key: str) -> tuple[bool, int]:
        return self._backend.allow(key)


class _InMemoryFixedWindowBackend:
    def __init__(self, max_requests: int, window_seconds: int, cleanup_interval_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._last_cleanup = time()
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
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


class _RedisFixedWindowBackend:
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        redis_url: str | None,
        key_prefix: str,
    ) -> None:
        if not redis_url:
            raise ValueError("redis_url is required when rate limiter backend is redis")

        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("redis backend requested but 'redis' package is not installed") from exc

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    def _bucket_key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def allow(self, key: str) -> tuple[bool, int]:
        bucket_key = self._bucket_key(key)
        count = self.client.incr(bucket_key)
        ttl = self.client.ttl(bucket_key)

        if count == 1 or ttl < 0:
            self.client.expire(bucket_key, self.window_seconds)
            ttl = self.window_seconds

        if count > self.max_requests:
            return False, max(1, int(ttl) if ttl and ttl > 0 else self.window_seconds)
        return True, 0
