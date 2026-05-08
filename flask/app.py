from __future__ import annotations

import json
import logging
import os
import sys
from threading import Lock
from time import perf_counter

from flask import Flask, Response, jsonify, request, stream_with_context

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Agent import Agent  # noqa: E402
from rate_limiter import FixedWindowRateLimiter  # noqa: E402
from session_agent_store import SessionAgentStore  # noqa: E402


def _configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        force=True,
    )
    root_logger = logging.getLogger()
    werkzeug_logger = logging.getLogger("werkzeug")
    flask_app_logger = logging.getLogger("plank-agent-flask")

    werkzeug_logger.setLevel(logging.INFO)
    werkzeug_logger.propagate = True

    flask_app_logger.setLevel(logging.INFO)
    flask_app_logger.propagate = True

    if not flask_app_logger.handlers:
        for handler in root_logger.handlers:
            flask_app_logger.addHandler(handler)

    return flask_app_logger


app = Flask(__name__)
logger = _configure_logging()
_startup_lock = Lock()
_startup_done = False


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _build_session_store() -> SessionAgentStore:
    return SessionAgentStore(
        ttl_seconds=_env_int("PLANK_SESSION_TTL_SECONDS", 1800),
        cleanup_interval_seconds=_env_int("PLANK_SESSION_CLEANUP_INTERVAL_SECONDS", 300),
        backend=_env_str("PLANK_SESSION_BACKEND", "memory"),
        redis_url=_env_str("PLANK_REDIS_URL", ""),
        key_prefix=_env_str("PLANK_SESSION_KEY_PREFIX", "plank-agent:session"),
        lock_timeout_seconds=_env_int("PLANK_SESSION_LOCK_TIMEOUT_SECONDS", 30),
        lock_blocking_timeout_seconds=_env_int("PLANK_SESSION_LOCK_BLOCKING_TIMEOUT_SECONDS", 15),
    )


def _build_limiter(key_prefix: str, max_requests: int) -> FixedWindowRateLimiter:
    return FixedWindowRateLimiter(
        max_requests=max_requests,
        window_seconds=_env_int("PLANK_RATE_LIMIT_WINDOW_SECONDS", 60),
        cleanup_interval_seconds=_env_int("PLANK_RATE_LIMIT_CLEANUP_INTERVAL_SECONDS", 300),
        backend=_env_str("PLANK_RATE_LIMIT_BACKEND", "memory"),
        redis_url=_env_str("PLANK_REDIS_URL", ""),
        key_prefix=key_prefix,
    )


agent_store = _build_session_store()
session_limiter = _build_limiter(
    key_prefix=_env_str("PLANK_SESSION_RATE_LIMIT_PREFIX", "plank-agent:ratelimit:session"),
    max_requests=_env_int("PLANK_SESSION_RATE_LIMIT_MAX_REQUESTS", 20),
)
ip_limiter = _build_limiter(
    key_prefix=_env_str("PLANK_IP_RATE_LIMIT_PREFIX", "plank-agent:ratelimit:ip"),
    max_requests=_env_int("PLANK_IP_RATE_LIMIT_MAX_REQUESTS", 100),
)


def _should_run_startup_hook() -> bool:
    if app.debug:
        return os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    return True


def _prewarm_agent_if_enabled() -> None:
    if not _env_bool("PLANK_AGENT_PREWARM", True):
        logger.info("agent prewarm disabled by PLANK_AGENT_PREWARM")
        return

    prewarm_session_id = os.getenv("PLANK_AGENT_PREWARM_SESSION_ID", "__prewarm__")
    started = perf_counter()
    try:
        Agent.prewarm()
        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "agent prewarm ok session_id=%s elapsed_ms=%s",
            prewarm_session_id,
            elapsed_ms,
        )
    except Exception:
        logger.exception("agent prewarm failed session_id=%s", prewarm_session_id)


def _trusted_proxy_ips() -> set[str]:
    raw = _env_str("PLANK_TRUSTED_PROXY_IPS", "")
    return {value.strip() for value in raw.split(",") if value.strip()}


def _client_ip() -> str:
    remote_addr = (request.remote_addr or "").strip() or "unknown"
    if not _env_bool("PLANK_TRUST_PROXY_HEADERS", False):
        return remote_addr

    trusted_proxies = _trusted_proxy_ips()
    if trusted_proxies and remote_addr not in trusted_proxies:
        return remote_addr

    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return client_ip
    return remote_addr


def _error(status_code: int, error: str, message: str):
    payload = {"error": error, "message": message}
    return jsonify(payload), status_code


def _check_rate_limit(session_id: str, ip: str):
    ok_session, retry_session = session_limiter.allow(f"sid:{session_id}")
    if not ok_session:
        resp = jsonify(
            {
                "error": "rate_limited",
                "message": "Too many requests for this session",
                "retry_after": retry_session,
            }
        )
        resp.status_code = 429
        resp.headers["Retry-After"] = str(retry_session)
        return resp

    ok_ip, retry_ip = ip_limiter.allow(f"ip:{ip}")
    if not ok_ip:
        resp = jsonify(
            {
                "error": "rate_limited",
                "message": "Too many requests from this IP",
                "retry_after": retry_ip,
            }
        )
        resp.status_code = 429
        resp.headers["Retry-After"] = str(retry_ip)
        return resp
    return None


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    logger.info("health check ip=%s status=200", _client_ip())
    return jsonify({"status": "ok"}), 200


def _run_startup_once() -> None:
    global _startup_done
    if _startup_done:
        return
    with _startup_lock:
        if _startup_done:
            return
        if _should_run_startup_hook():
            _prewarm_agent_if_enabled()
        _startup_done = True


@app.before_request
def startup() -> None:
    _run_startup_once()


@app.post("/chat")
def chat():
    started = perf_counter()
    session_id = request.headers.get("X-Session-Id", "").strip()
    ip = _client_ip()

    logger.info(
        "chat request received session_id=%s ip=%s method=%s path=%s",
        session_id or "(missing)",
        ip,
        request.method,
        request.path,
    )

    if not session_id:
        return _error(400, "missing_session_id", "X-Session-Id is required")

    limited = _check_rate_limit(session_id, ip)
    if limited is not None:
        return limited

    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    if not user_message:
        return _error(400, "invalid_request", "JSON field 'message' is required")

    want_stream = _as_bool(payload.get("stream"), default=False)
    include_memory = _as_bool(
        payload.get("include_memory"),
        default=_env_bool("PLANK_WEB_INCLUDE_MEMORY", True),
    )
    persist_memory = _as_bool(
        payload.get("persist_memory"),
        default=_env_bool("PLANK_WEB_PERSIST_MEMORY", True),
    )

    if not want_stream:
        try:
            with agent_store.session_lock(session_id):
                agent = Agent(name="PlankAgent", user_id=session_id)
                agent.restore_session_state(agent_store.load_messages(session_id))
                answer = agent.run(
                    user_input=user_message,
                    return_trace=False,
                    silent=True,
                    include_memory=include_memory,
                    persist_memory=persist_memory,
                )
                agent_store.save_messages(session_id, agent.export_session_state())
        except Exception as exc:
            logger.exception("chat failed session_id=%s ip=%s", session_id, ip)
            return _error(500, "internal_error", f"Chat failed: {exc}")

        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "chat ok session_id=%s ip=%s elapsed_ms=%s status=200 stream=%s memory=%s persist_memory=%s",
            session_id,
            ip,
            elapsed_ms,
            False,
            include_memory,
            persist_memory,
        )
        return jsonify({"session_id": session_id, "answer": answer, "elapsed_ms": elapsed_ms}), 200

    @stream_with_context
    def event_stream():
        chunks = 0
        first_token_ms: int | None = None
        try:
            yield ": stream-start\n\n"
            with agent_store.session_lock(session_id):
                agent = Agent(name="PlankAgent", user_id=session_id)
                agent.restore_session_state(agent_store.load_messages(session_id))
                for delta in agent.run_stream(
                    user_input=user_message,
                    include_memory=include_memory,
                    persist_memory=persist_memory,
                ):
                    if first_token_ms is None:
                        first_token_ms = int((perf_counter() - started) * 1000)
                    chunks += 1
                    yield _sse_event("delta", {"text": delta})
                agent_store.save_messages(session_id, agent.export_session_state())

            elapsed_ms = int((perf_counter() - started) * 1000)
            logger.info(
                "chat ok session_id=%s ip=%s elapsed_ms=%s first_token_ms=%s status=200 stream=%s chunks=%s memory=%s persist_memory=%s",
                session_id,
                ip,
                elapsed_ms,
                first_token_ms,
                True,
                chunks,
                include_memory,
                persist_memory,
            )
            yield _sse_event(
                "done",
                {
                    "session_id": session_id,
                    "elapsed_ms": elapsed_ms,
                    "first_token_ms": first_token_ms,
                },
            )
        except Exception as exc:
            logger.exception("chat stream failed session_id=%s ip=%s", session_id, ip)
            yield _sse_event("error", {"error": "internal_error", "message": f"Chat failed: {exc}"})

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(event_stream(), headers=headers)


if __name__ == "__main__":
    _run_startup_once()
    app.run(host="0.0.0.0", port=6543)
