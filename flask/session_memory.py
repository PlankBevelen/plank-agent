from __future__ import annotations

from threading import Lock
from time import time


class SingleTurnMemoryStore:
    """
    Keep only the latest turn for each session id in memory.
    """

    def __init__(self, ttl_seconds: int = 1800, cleanup_interval_seconds: int = 300) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._store: dict[str, dict[str, str | float]] = {}
        self._last_cleanup = time()
        self._lock = Lock()

    def get_last_turn(self, session_id: str) -> dict[str, str] | None:
        now = time()
        with self._lock:
            self._cleanup_if_needed(now)
            data = self._store.get(session_id)
            if not data:
                return None
            if now - float(data.get("updated_at", now)) > self.ttl_seconds:
                self._store.pop(session_id, None)
                return None
            return {
                "last_user_message": str(data.get("last_user_message", "")),
                "last_assistant_message": str(data.get("last_assistant_message", "")),
            }

    def set_last_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        now = time()
        with self._lock:
            self._cleanup_if_needed(now)
            self._store[session_id] = {
                "last_user_message": user_message,
                "last_assistant_message": assistant_message,
                "updated_at": now,
            }

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
