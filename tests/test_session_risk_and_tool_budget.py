"""
Cross-process tool-call budget enforcement, and cumulative-risk flagging,
against all three backends. test_end_to_end.py exercises record_turn_risk/
is_session_flagged/reset_session too, but only for Redis (it's Redis-only,
since it's about the full request flow, not about backend technology) —
these close that gap for PostgreSQL and MySQL, and check_tool_call/
increment_tool_calls aren't exercised by any other test for any backend.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor

import pytest

from llm_security_pipeline import RateLimitExceeded, SessionLimits, SessionRateLimiter

from conftest import MYSQL_DB, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER, POSTGRES_DSN, REDIS_URL


def _worker_make_tool_call_redis(session_id: str, limits_kwargs: dict) -> str:
    from llm_security_pipeline import RedisStateBackend

    async def run():
        backend = RedisStateBackend.from_url(REDIS_URL)
        limiter = SessionRateLimiter(limits=SessionLimits(**limits_kwargs), store=backend.session_store)
        try:
            await limiter.check_tool_call(session_id)
            return "ALLOWED"
        except RateLimitExceeded as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


def _worker_make_tool_call_postgres(session_id: str, limits_kwargs: dict) -> str:
    from llm_security_pipeline import PostgresStateBackend

    async def run():
        backend = await PostgresStateBackend.create(dsn=POSTGRES_DSN)
        limiter = SessionRateLimiter(limits=SessionLimits(**limits_kwargs), store=backend.session_store)
        try:
            await limiter.check_tool_call(session_id)
            return "ALLOWED"
        except RateLimitExceeded as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


def _worker_make_tool_call_mysql(session_id: str, limits_kwargs: dict) -> str:
    from llm_security_pipeline import MySQLStateBackend

    async def run():
        backend = await MySQLStateBackend.create(
            host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB,
        )
        limiter = SessionRateLimiter(limits=SessionLimits(**limits_kwargs), store=backend.session_store)
        try:
            await limiter.check_tool_call(session_id)
            return "ALLOWED"
        except RateLimitExceeded as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


_TOOL_CALL_WORKERS = {
    "redis": _worker_make_tool_call_redis,
    "postgres": _worker_make_tool_call_postgres,
    "mysql": _worker_make_tool_call_mysql,
}


async def _build_backend(backend_name: str):
    if backend_name == "redis":
        from llm_security_pipeline import RedisStateBackend

        return RedisStateBackend.from_url(REDIS_URL)
    if backend_name == "postgres":
        from llm_security_pipeline import PostgresStateBackend

        return await PostgresStateBackend.create(dsn=POSTGRES_DSN)
    if backend_name == "mysql":
        from llm_security_pipeline import MySQLStateBackend

        return await MySQLStateBackend.create(
            host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB,
        )
    raise ValueError(backend_name)


_BACKEND_PARAMS = [
    pytest.param("redis", marks=pytest.mark.redis),
    pytest.param("postgres", marks=pytest.mark.postgres),
    pytest.param("mysql", marks=pytest.mark.mysql),
]


@pytest.mark.integration
@pytest.mark.parametrize("backend_name", _BACKEND_PARAMS)
async def test_tool_call_budget_enforced_across_processes(backend_name):
    worker = _TOOL_CALL_WORKERS[backend_name]
    session_id = f"test-tool-call-budget-{backend_name}"
    limits_kwargs = dict(window_seconds=60, max_tool_calls_per_window=2)

    backend = await _build_backend(backend_name)
    await backend.session_store.reset_session(session_id)  # clean slate for repeatable runs
    await backend.aclose()

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = [
            loop.run_in_executor(pool, worker, session_id, limits_kwargs)
            for _ in range(5)
        ]
        results = await asyncio.gather(*futures)

    allowed = sum(1 for r in results if r == "ALLOWED")
    assert allowed == 2, (
        f"[{backend_name}] expected exactly 2 of 5 racing processes to be allowed (budget=2), got: {results}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("backend_name", _BACKEND_PARAMS)
async def test_cumulative_risk_flagging_per_backend(backend_name):
    session_id = f"test-cumulative-risk-{backend_name}"
    limits = SessionLimits(cumulative_risk_threshold=1.0, risk_decay_per_second=0.0)

    backend = await _build_backend(backend_name)
    try:
        limiter = SessionRateLimiter(limits=limits, store=backend.session_store)
        await limiter.reset_session(session_id)

        assert await limiter.is_session_flagged(session_id) is False

        cumulative = await limiter.record_turn_risk(session_id, 0.6)
        assert cumulative == pytest.approx(0.6)
        assert await limiter.is_session_flagged(session_id) is False

        cumulative = await limiter.record_turn_risk(session_id, 0.6)
        assert cumulative == pytest.approx(1.2)
        assert await limiter.is_session_flagged(session_id) is True

        await limiter.reset_session(session_id)
        assert await limiter.is_session_flagged(session_id) is False
    finally:
        await backend.aclose()