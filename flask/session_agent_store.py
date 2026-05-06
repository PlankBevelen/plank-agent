from __future__ import annotations

from threading import Lock
from time import time
from typing import Any, Callable


class SessionAgentStore:
    """
    Keep one Agent instance per session id with TTL-based eviction.
    Returns (agent, session_lock) so callers can serialize same-session calls.
    """

    def __init__(
        self,
        factory: Callable[[str], Any],
        ttl_seconds: int = 1800,
        cleanup_interval_seconds: int = 300,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self.factory = factory
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._store: dict[str, dict[str, Any]] = {}
        self._last_cleanup = time()
        self._lock = Lock()

    def get_or_create(self, session_id: str):
        now = time()
        with self._lock:
            self._cleanup_if_needed(now)
            payload = self._store.get(session_id)
            if payload is not None:
                if now - float(payload.get("updated_at", now)) <= self.ttl_seconds:
                    payload["updated_at"] = now
                    return payload["agent"], payload["lock"]
                self._store.pop(session_id, None)

            agent = self.factory(session_id)
            session_lock = Lock()
            self._store[session_id] = {
                "agent": agent,
                "lock": session_lock,
                "updated_at": now,
            }
            return agent, session_lock

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
        self._last_cleanup = now
