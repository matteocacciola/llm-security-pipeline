"""
Cross-process capability-token replay safety.

Spawns REAL separate OS processes (not just asyncio tasks in one process)
that all race to redeem the same single-use token. This is the scenario an
in-memory nonce store cannot protect against (each process would keep its
own private counter), and it's run against all three backends: each is
supposed to provide the exact same guarantee, on different technology.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor

import pytest

from llm_security_pipeline import CapabilityToken, ScopeError, ScopeGuard

from conftest import MYSQL_DB, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER, POSTGRES_DSN, REDIS_URL


def _worker_redeem_token_redis(secret_key_hex: str, token_payload: dict, token_signature: str, action: str) -> str:
    """Runs in its own OS process: builds its own RedisStateBackend and
    tries to redeem the given token. Simulates one pod/worker among many
    all handling requests that reference the same capability token."""
    from llm_security_pipeline import RedisStateBackend

    async def run():
        backend = RedisStateBackend.from_url(REDIS_URL)
        guard = ScopeGuard(secret_key=bytes.fromhex(secret_key_hex), nonce_store=backend.nonce_store)
        token = CapabilityToken(payload=token_payload, signature=token_signature)
        try:
            await guard.authorize(token, action)
            return "ALLOWED"
        except ScopeError as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


def _worker_redeem_token_postgres(secret_key_hex: str, token_payload: dict, token_signature: str, action: str) -> str:
    from llm_security_pipeline import PostgresStateBackend

    async def run():
        backend = await PostgresStateBackend.create(dsn=POSTGRES_DSN)
        guard = ScopeGuard(secret_key=bytes.fromhex(secret_key_hex), nonce_store=backend.nonce_store)
        token = CapabilityToken(payload=token_payload, signature=token_signature)
        try:
            await guard.authorize(token, action)
            return "ALLOWED"
        except ScopeError as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


def _worker_redeem_token_mysql(secret_key_hex: str, token_payload: dict, token_signature: str, action: str) -> str:
    from llm_security_pipeline import MySQLStateBackend

    async def run():
        backend = await MySQLStateBackend.create(
            host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB,
        )
        guard = ScopeGuard(secret_key=bytes.fromhex(secret_key_hex), nonce_store=backend.nonce_store)
        token = CapabilityToken(payload=token_payload, signature=token_signature)
        try:
            await guard.authorize(token, action)
            return "ALLOWED"
        except ScopeError as exc:
            return f"DENIED ({exc})"
        finally:
            await backend.aclose()

    return asyncio.run(run())


_WORKERS = {
    "redis": _worker_redeem_token_redis,
    "postgres": _worker_redeem_token_postgres,
    "mysql": _worker_redeem_token_mysql,
}


@pytest.mark.integration
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param("redis", marks=pytest.mark.redis),
        pytest.param("postgres", marks=pytest.mark.postgres),
        pytest.param("mysql", marks=pytest.mark.mysql),
    ],
)
async def test_single_use_token_redeemed_exactly_once_across_processes(backend_name):
    worker = _WORKERS[backend_name]

    guard = ScopeGuard()  # secret_key generated here, shared explicitly with workers below
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"], ttl_seconds=30, max_uses=1)
    secret_key_hex = guard._secret_key.hex()  # noqa: SLF001 - test needs to share the key across processes

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = [
            loop.run_in_executor(
                pool, worker, secret_key_hex, token.payload, token.signature, "read_crm",
            )
            for _ in range(5)
        ]
        results = await asyncio.gather(*futures)

    allowed = sum(1 for r in results if r == "ALLOWED")
    assert allowed == 1, (
        f"[{backend_name}] expected exactly 1 of 5 racing processes to redeem a max_uses=1 token, got: {results}"
    )