"""
Fixtures shared by the integration test suite.

Tests marked @pytest.mark.integration exercise cross-process correctness
(real OS processes racing against a real shared-state backend), not pure
single-process logic. Three backends exist (Redis, PostgreSQL, MySQL);
each cross-process test is parametrized over all three and additionally
marked with the matching `redis`/`postgres`/`mysql` marker, so a given
backend's cases skip automatically when that backend isn't reachable
(missing driver package or nothing listening), independently of the
others. Point the relevant env vars at a running instance to exercise it;
see each backend's connection constants below for the defaults and the
env vars that override them.
"""

from __future__ import annotations

import asyncio
import os

import pytest
import redis.asyncio as redis_async

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# A cluster-enabled server, used only by the Redis Cluster key-layout tests.
# One node owning all 16384 slots is enough: CROSSSLOT is rejected by the
# client and the server regardless of how many nodes hold the slots, so a
# full multi-node cluster would slow CI down without testing anything more.
REDIS_CLUSTER_URL = os.environ.get("REDIS_CLUSTER_URL", "redis://127.0.0.1:7000")

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")

MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DB = os.environ.get("MYSQL_DB", "test")


def _redis_reachable() -> bool:
    async def _ping() -> bool:
        client = redis_async.from_url(REDIS_URL)
        try:
            await client.ping()
            return True
        except Exception:
            return False
        finally:
            await client.aclose()

    return asyncio.run(_ping())


def _redis_cluster_reachable() -> bool:
    async def _ping() -> bool:
        try:
            from redis.asyncio.cluster import RedisCluster
        except ImportError:
            return False
        try:
            client = RedisCluster.from_url(REDIS_CLUSTER_URL)
        except Exception:
            return False
        try:
            # Reachability is not enough: a node with unassigned slots
            # accepts connections but refuses every keyed command.
            info = await client.cluster_info()
            return info.get("cluster_state") == "ok"
        except Exception:
            return False
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    return asyncio.run(_ping())


def _postgres_reachable() -> bool:
    try:
        import asyncpg
    except ImportError:
        return False

    async def _ping() -> bool:
        try:
            conn = await asyncpg.connect(POSTGRES_DSN, timeout=3)
        except Exception:
            return False
        try:
            await conn.execute("SELECT 1")
            return True
        finally:
            await conn.close()

    return asyncio.run(_ping())


def _mysql_reachable() -> bool:
    try:
        import aiomysql
    except ImportError:
        return False

    async def _ping() -> bool:
        try:
            conn = await asyncio.wait_for(
                aiomysql.connect(
                    host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB,
                ),
                timeout=3,
            )
        except Exception:
            return False
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
            return True
        finally:
            conn.close()

    return asyncio.run(_ping())


# Backend name -> (reachability check, human-readable connection target).
# Each name doubles as the pytest marker used on the matching test cases.
_BACKEND_CHECKS = {
    "redis": (_redis_reachable, REDIS_URL),
    "redis_cluster": (_redis_cluster_reachable, REDIS_CLUSTER_URL),
    "postgres": (_postgres_reachable, POSTGRES_DSN),
    "mysql": (_mysql_reachable, f"{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    unreachable = {
        name: target for name, (check, target) in _BACKEND_CHECKS.items() if not check()
    }
    if not unreachable:
        return
    for item in items:
        for name, target in unreachable.items():
            if name in item.keywords:
                item.add_marker(pytest.mark.skip(reason=f"{name} backend not reachable at {target}"))
