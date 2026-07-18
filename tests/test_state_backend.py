"""
Tests for StateBackend/RedisStateBackend/PostgresStateBackend/
MySQLStateBackend: the ownership contract (aclose() must only release a
connection/pool the backend itself created, never one the caller passed
in and still needs) and the input-validation error paths, none of which
any other test exercises.
"""

from __future__ import annotations

import pytest

from llm_security_pipeline import (
    InMemoryNonceStore,
    InMemorySessionStore,
    MySQLStateBackend,
    PostgresStateBackend,
    StateBackend,
)

from conftest import MYSQL_DB, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER, POSTGRES_DSN, REDIS_URL


async def test_state_backend_base_aclose_is_a_noop():
    backend = StateBackend(nonce_store=InMemoryNonceStore(), session_store=InMemorySessionStore())
    await backend.aclose()  # must not raise; the base bundle owns no connection to release


def test_postgres_state_backend_create_requires_dsn_or_pool():
    pytest.importorskip("asyncpg")

    import asyncio

    with pytest.raises(ValueError, match="dsn or an existing pool"):
        asyncio.run(PostgresStateBackend.create())


def test_mysql_state_backend_create_requires_pool_or_kwargs():
    pytest.importorskip("aiomysql")

    import asyncio

    with pytest.raises(ValueError, match="connection kwargs .* or an existing pool"):
        asyncio.run(MySQLStateBackend.create())


@pytest.mark.integration
@pytest.mark.redis
async def test_redis_state_backend_does_not_close_a_client_it_does_not_own():
    from llm_security_pipeline import RedisStateBackend
    import redis.asyncio as redis_async

    client = redis_async.from_url(REDIS_URL)
    try:
        backend = RedisStateBackend(client)  # owns_client defaults to False
        await backend.aclose()

        # The caller's own client must still be usable after the backend closes.
        assert await client.ping() is True
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.postgres
async def test_postgres_state_backend_does_not_close_a_pool_it_does_not_own():
    asyncpg = pytest.importorskip("asyncpg")

    pool = await asyncpg.create_pool(POSTGRES_DSN)
    try:
        backend = await PostgresStateBackend.create(pool=pool)
        await backend.aclose()

        # The caller's own pool must still be usable after the backend closes.
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT 1") == 1
    finally:
        await pool.close()


@pytest.mark.integration
@pytest.mark.mysql
async def test_mysql_state_backend_does_not_close_a_pool_it_does_not_own():
    aiomysql = pytest.importorskip("aiomysql")

    pool = await aiomysql.create_pool(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB,
    )
    try:
        backend = await MySQLStateBackend.create(pool=pool)
        await backend.aclose()

        # The caller's own pool must still be usable after the backend closes.
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                (value,) = await cur.fetchone()
                assert value == 1
    finally:
        pool.close()
        await pool.wait_closed()