from __future__ import annotations

import json
from contextlib import contextmanager
from threading import Lock
from time import time
from typing import Iterator


class SessionAgentStore:
    def __init__(
        self,
        ttl_seconds: int = 1800,
        cleanup_interval_seconds: int = 300,
        backend: str = "memory",
        redis_url: str | None = None,
        key_prefix: str = "plank-agent:session",
        lock_timeout_seconds: int = 30,
        lock_blocking_timeout_seconds: int = 15,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

        backend = (backend or "memory").strip().lower()
        if backend == "redis":
            self._backend = _RedisSessionBackend(
                redis_url=redis_url,
                ttl_seconds=ttl_seconds,
                key_prefix=key_prefix,
                lock_timeout_seconds=lock_timeout_seconds,
                lock_blocking_timeout_seconds=lock_blocking_timeout_seconds,
            )
        else:
            self._backend = _InMemorySessionBackend(
                ttl_seconds=ttl_seconds,
                cleanup_interval_seconds=cleanup_interval_seconds,
            )

    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        return self._backend.load_messages(session_id)

    def save_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        self._backend.save_messages(session_id, messages)

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[None]:
        with self._backend.session_lock(session_id):
            yield


class _InMemorySessionBackend:
    def __init__(self, ttl_seconds: int, cleanup_interval_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._store: dict[str, dict[str, object]] = {}
        self._locks: dict[str, Lock] = {}
        self._last_cleanup = time()
        self._lock = Lock()

    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        now = time()
        with self._lock:
            self._cleanup_if_needed(now)
            payload = self._store.get(session_id)
            if payload is None:
                return []
            if now - float(payload.get("updated_at", now)) > self.ttl_seconds:
                self._store.pop(session_id, None)
                return []
            payload["updated_at"] = now
            messages = payload.get("messages") or []
            return [dict(item) for item in messages if isinstance(item, dict)]

    def save_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        now = time()
        with self._lock:
            self._cleanup_if_needed(now)
            self._store[session_id] = {
                "messages": [dict(item) for item in messages],
                "updated_at": now,
            }

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[None]:
        with self._lock:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = Lock()
                self._locks[session_id] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _cleanup_if_needed(self, now: float) -> None:
        if now - self._last_cleanup < self.cleanup_interval_seconds:
            return

        expired = [
            sid
            for sid, payload in self._store.items()
            if now - float(payload.get("updated_at", now)) > self.ttl_seconds
        ]
        for sid in expired:
            self._store.pop(sid, None)
            self._locks.pop(sid, None)
        self._last_cleanup = now


class _RedisSessionBackend:
    def __init__(
        self,
        redis_url: str | None,
        ttl_seconds: int,
        key_prefix: str,
        lock_timeout_seconds: int,
        lock_blocking_timeout_seconds: int,
    ) -> None:
        if not redis_url:
            raise ValueError("redis_url is required when session backend is redis")

        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("redis backend requested but 'redis' package is not installed") from exc

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.lock_timeout_seconds = lock_timeout_seconds
        self.lock_blocking_timeout_seconds = lock_blocking_timeout_seconds

    def _session_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:state:{session_id}"

    def _lock_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:lock:{session_id}"

    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        raw = self.client.get(self._session_key(session_id))
        if not raw:
            return []
        self.client.expire(self._session_key(session_id), self.ttl_seconds)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [dict(item) for item in payload if isinstance(item, dict)]

    def save_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        self.client.set(
            self._session_key(session_id),
            json.dumps(messages, ensure_ascii=False),
            ex=self.ttl_seconds,
        )

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[None]:
        lock = self.client.lock(
            self._lock_key(session_id),
            timeout=self.lock_timeout_seconds,
            blocking_timeout=self.lock_blocking_timeout_seconds,
            thread_local=False,
        )
        with lock:
            yield
