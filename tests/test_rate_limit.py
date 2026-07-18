"""
Cross-process session request-budget enforcement.

Same idea as the token-replay test, applied to the per-session request
budget: proves the limit holds even when traffic is spread across
independent processes/pods, which an in-memory SessionStore cannot
guarantee. Run against all three backends: each is supposed to provide
the exact same guarantee, on different technology.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor

import pytest

from llm_security_pipeline import RateLimitExceeded, SessionLimits, SessionRateLimiter

from conftest import MYSQL_DB, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER, POSTGRES_DSN, REDIS_URL


def _worker_make_request_redis(session_id: str, limits_kwargs: dict) -> str:
    """Runs in its own OS process: builds its own RedisStateBackend and
    independently hits the shared request budget for the same session_id."""
    from llm_security_pipeline import RedisStateBackend

    async def run():
        backend = RedisStateBackend.from_url(REDIS_URL)
        limiter = SessionRateLimiter(limits=SessionLimits(**limits_kwargs), store=backend.session_store)
        try:
            await limiter.check_request(session_id)
            return "ALLOWED"
        except RateLimitExceeded as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


def _worker_make_request_postgres(session_id: str, limits_kwargs: dict) -> str:
    from llm_security_pipeline import PostgresStateBackend

    async def run():
        backend = await PostgresStateBackend.create(dsn=POSTGRES_DSN)
        limiter = SessionRateLimiter(limits=SessionLimits(**limits_kwargs), store=backend.session_store)
        try:
            await limiter.check_request(session_id)
            return "ALLOWED"
        except RateLimitExceeded as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


def _worker_make_request_mysql(session_id: str, limits_kwargs: dict) -> str:
    from llm_security_pipeline import MySQLStateBackend

    async def run():
        backend = await MySQLStateBackend.create(
            host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB,
        )
        limiter = SessionRateLimiter(limits=SessionLimits(**limits_kwargs), store=backend.session_store)
        try:
            await limiter.check_request(session_id)
            return "ALLOWED"
        except RateLimitExceeded as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


_WORKERS = {
    "redis": _worker_make_request_redis,
    "postgres": _worker_make_request_postgres,
    "mysql": _worker_make_request_mysql,
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


@pytest.mark.integration
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param("redis", marks=pytest.mark.redis),
        pytest.param("postgres", marks=pytest.mark.postgres),
        pytest.param("mysql", marks=pytest.mark.mysql),
    ],
)
async def test_request_budget_enforced_across_processes(backend_name):
    worker = _WORKERS[backend_name]
    session_id = f"test-cross-process-session-{backend_name}"
    limits_kwargs = dict(window_seconds=60, max_requests_per_window=3)

    backend = await _build_backend(backend_name)
    await backend.session_store.reset_session(session_id)  # clean slate for repeatable runs
    await backend.aclose()

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [
            loop.run_in_executor(pool, worker, session_id, limits_kwargs)
            for _ in range(6)
        ]
        results = await asyncio.gather(*futures)

    allowed = sum(1 for r in results if r == "ALLOWED")
    assert allowed == 3, (
        f"[{backend_name}] expected exactly 3 of 6 racing processes to be allowed (budget=3), got: {results}"
    )