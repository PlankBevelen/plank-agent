from __future__ import annotations

import json
import logging
import os
import sys
from time import perf_counter

from flask import Flask, Response, jsonify, request, stream_with_context

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Agent import Agent  # noqa: E402
from rate_limiter import FixedWindowRateLimiter  # noqa: E402
from session_agent_store import SessionAgentStore  # noqa: E402

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plank-agent-flask")

agent_store = SessionAgentStore(
    factory=lambda session_id: Agent(name="PlankAgent", user_id=session_id),
    ttl_seconds=1800,
)
session_limiter = FixedWindowRateLimiter(max_requests=20, window_seconds=60)
ip_limiter = FixedWindowRateLimiter(max_requests=100, window_seconds=60)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _should_run_startup_hook() -> bool:
    # In debug reloader mode, only run startup logic in the serving process.
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
        agent_store.get_or_create(prewarm_session_id)
        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "agent prewarm ok session_id=%s elapsed_ms=%s",
            prewarm_session_id,
            elapsed_ms,
        )
    except Exception:
        logger.exception("agent prewarm failed session_id=%s", prewarm_session_id)


def _client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


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
    return jsonify({"status": "ok"}), 200


@app.before_serving
def startup() -> None:
    if not _should_run_startup_hook():
        return
    _prewarm_agent_if_enabled()


@app.post("/chat")
def chat():
    started = perf_counter()
    session_id = request.headers.get("X-Session-Id", "").strip()
    ip = _client_ip()

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
    agent, session_lock = agent_store.get_or_create(session_id)

    if not want_stream:
        try:
            # Serialize calls per session to protect shared agent state.
            with session_lock:
                answer = agent.run(
                    user_input=user_message,
                    return_trace=False,
                    silent=True,
                    include_memory=False,
                    persist_memory=False,
                )
        except Exception as exc:
            logger.exception("chat failed session_id=%s ip=%s", session_id, ip)
            return _error(500, "internal_error", f"Chat failed: {exc}")

        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "chat ok session_id=%s ip=%s elapsed_ms=%s status=200 stream=%s",
            session_id,
            ip,
            elapsed_ms,
            False,
        )
        return jsonify({"session_id": session_id, "answer": answer, "elapsed_ms": elapsed_ms}), 200

    @stream_with_context
    def event_stream():
        chunks = 0
        try:
            with session_lock:
                for delta in agent.run_stream(
                    user_input=user_message,
                    include_memory=False,
                    persist_memory=False,
                ):
                    chunks += 1
                    yield _sse_event("delta", {"text": delta})

            elapsed_ms = int((perf_counter() - started) * 1000)
            logger.info(
                "chat ok session_id=%s ip=%s elapsed_ms=%s status=200 stream=%s chunks=%s",
                session_id,
                ip,
                elapsed_ms,
                True,
                chunks,
            )
            yield _sse_event("done", {"session_id": session_id, "elapsed_ms": elapsed_ms})
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
    app.run(host="0.0.0.0", port=6543)
