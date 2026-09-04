"""
state_backend.py
Backend-neutral bundle of the shared-state components the pipeline needs
in a multi-process deployment: a NonceStore (capability-token replay
tracking), a SessionStore (rate limits + cumulative risk), and optionally
an audit logger.

Why this exists: the pipeline's cross-process correctness depends on a
PROPERTY — a centralized store with atomic check-and-update operations —
not on a specific PRODUCT. Redis is one good implementation of that
property; PostgreSQL, MySQL or DynamoDB can implement it equally well
(atomic upserts / conditional writes / transactions). Naming the pipeline
constructor parameters after Redis suggested otherwise, so the facade now
takes a generic `state_backend` instead, and Redis is provided as just one
ready-made implementation of it.

Usage:

    # Redis (ready-made)
    backend = RedisStateBackend.from_url("redis://redis-service:6379/0")

    # Redis Cluster
    backend = RedisStateBackend.from_cluster_url("redis://redis-cluster:6379")
    pipeline = SecurityPipeline(state_backend=backend, ...)

    # Any other technology: implement the two store interfaces from
    # stores.py and bundle them:
    backend = StateBackend(
        nonce_store=PostgresNonceStore(pool),
        session_store=PostgresSessionStore(pool),
    )
    pipeline = SecurityPipeline(state_backend=backend, ...)

    # No backend at all -> in-memory stores, valid for single-process use.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sessions.stores import NonceStore, ProvenanceStore, SessionStore

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover - redis is an optional dependency
    Redis = None  # type: ignore


@dataclass
class StateBackend:
    """A bundle of shared-state stores backing the pipeline's cross-process
    guarantees. Build one from any technology by implementing NonceStore
    and SessionStore; both interfaces are small (one and five methods
    respectively) and only require that check-and-update operations be
    atomic on the backing store."""

    nonce_store: NonceStore
    session_store: SessionStore
    # Optional third store, needed only for ingest-time provenance
    # (services/ingest_guard.py). Left None rather than defaulted to an
    # in-memory instance on purpose: a provenance store that forgets on
    # restart turns verified retrievals into unverified ones without
    # anything failing, which is exactly the kind of silent downgrade this
    # library tries not to ship.
    provenance_store: ProvenanceStore | None = None

    async def aclose(self) -> None:
        """Override in subclasses that own an underlying connection/pool
        and need to release it on shutdown. The base bundle owns nothing."""


class RedisStateBackend(StateBackend):
    """Ready-made StateBackend on Redis: atomic Lua scripts for replay
    tracking and rate/risk counters. See redis_stores.py for details.

    Works on a single node and on Redis Cluster. On a cluster the keys a
    script touches together must share a hash slot, so session keys are
    wrapped in a hash tag; the topology is detected on first use, and
    `hash_tags` (True / False / "auto") overrides that decision. Use
    `from_cluster_url` to have the backend build the cluster client too."""

    def __init__(
        self,
        redis_client: "Redis",
        key_prefix: str = "sentinel:",
        owns_client: bool = False,
        hash_tags: bool | str = "auto",
    ):
        from .sessions.redis_stores import RedisNonceStore, RedisSessionStore

        if Redis is None:
            raise RuntimeError(
                "The 'redis' package is required for RedisStateBackend. "
                "Install with: pip install 'llm-security-pipeline[redis]'"
            )
        from .sessions.redis_stores import RedisProvenanceStore

        super().__init__(
            nonce_store=RedisNonceStore(redis_client, key_prefix=f"{key_prefix}nonce:"),
            session_store=RedisSessionStore(
                redis_client, key_prefix=f"{key_prefix}session:", hash_tags=hash_tags,
            ),
            provenance_store=RedisProvenanceStore(
                redis_client, key_prefix=f"{key_prefix}provenance:",
            ),
        )
        self.redis_client = redis_client
        self._owns_client = owns_client

    @classmethod
    def from_url(
        cls, url: str, key_prefix: str = "sentinel:", hash_tags: bool | str = "auto",
    ) -> "RedisStateBackend":
        """Create a backend that owns its own client (closed by aclose()).
        Prefer passing an existing shared client to __init__ when your
        application already manages one."""
        if Redis is None:
            raise RuntimeError(
                "The 'redis' package is required for RedisStateBackend. "
                "Install with: pip install 'llm-security-pipeline[redis]'"
            )
        return cls(Redis.from_url(url), key_prefix=key_prefix, owns_client=True, hash_tags=hash_tags)

    @classmethod
    def from_cluster_url(
        cls, url: str, key_prefix: str = "sentinel:", **cluster_kwargs,
    ) -> "RedisStateBackend":
        """Same, against a Redis Cluster: builds a RedisCluster client that
        discovers the topology from `url` and routes each command to the
        node owning its slot. Session keys are hash-tagged automatically."""
        try:
            from redis.asyncio.cluster import RedisCluster
        except ImportError as exc:
            raise RuntimeError(
                "The 'redis' package (>=5) is required for RedisStateBackend. "
                "Install with: pip install 'llm-security-pipeline[redis]'"
            ) from exc
        return cls(
            # RedisCluster is not a subclass of Redis in the stubs, but it
            # implements every command this backend issues.
            RedisCluster.from_url(url, **cluster_kwargs),  # type: ignore[arg-type]
            key_prefix=key_prefix,
            owns_client=True,
            hash_tags=True,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.redis_client.aclose()


class PostgresStateBackend(StateBackend):
    """Ready-made StateBackend on PostgreSQL: atomic single-statement
    upserts (INSERT ... ON CONFLICT ... RETURNING) for replay tracking and
    rate/risk counters. See postgres_stores.py for details.

    Requires the 'postgres' extra (asyncpg). Use `create(...)` — schema
    setup is part of construction:

        backend = await PostgresStateBackend.create("postgresql://user:pw@host/db")
        # or with an existing pool your application manages:
        backend = await PostgresStateBackend.create(pool=my_pool)
    """

    def __init__(self, pool, owns_pool: bool = False):
        from .sessions.postgres_stores import (
            PostgresNonceStore,
            PostgresProvenanceStore,
            PostgresSessionStore,
        )

        super().__init__(
            nonce_store=PostgresNonceStore(pool),
            session_store=PostgresSessionStore(pool),
            provenance_store=PostgresProvenanceStore(pool),
        )
        self.pool = pool
        self._owns_pool = owns_pool

    @classmethod
    async def create(cls, dsn: str | None = None, pool=None) -> "PostgresStateBackend":
        from .sessions.postgres_stores import ensure_schema

        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError(
                "The 'asyncpg' package is required for PostgresStateBackend. "
                "Install with: uv add 'llm-security-pipeline[postgres]'"
            ) from exc
        owns_pool = pool is None
        if pool is None:
            if dsn is None:
                raise ValueError("Provide either a dsn or an existing pool.")
            pool = await asyncpg.create_pool(dsn)
        await ensure_schema(pool)
        return cls(pool, owns_pool=owns_pool)

    async def aclose(self) -> None:
        if self._owns_pool:
            await self.pool.close()


class MySQLStateBackend(StateBackend):
    """Ready-made StateBackend on MySQL/MariaDB: short transactions with
    SELECT ... FOR UPDATE row locks for replay tracking and rate/risk
    counters. See mysql_stores.py for details.

    Requires the 'mysql' extra (aiomysql). Use `create(...)`:

        backend = await MySQLStateBackend.create(
            host="db-host", user="sentinel", password="...", db="sentinel",
        )
        # or with an existing pool your application manages:
        backend = await MySQLStateBackend.create(pool=my_pool)
    """

    def __init__(self, pool, owns_pool: bool = False):
        from .sessions.mysql_stores import (
            MySQLNonceStore,
            MySQLProvenanceStore,
            MySQLSessionStore,
        )

        super().__init__(
            nonce_store=MySQLNonceStore(pool),
            session_store=MySQLSessionStore(pool),
            provenance_store=MySQLProvenanceStore(pool),
        )
        self.pool = pool
        self._owns_pool = owns_pool

    @classmethod
    async def create(cls, pool=None, **aiomysql_connect_kwargs) -> "MySQLStateBackend":
        from .sessions.mysql_stores import ensure_schema

        try:
            import aiomysql
        except ImportError as exc:
            raise RuntimeError(
                "The 'aiomysql' package is required for MySQLStateBackend. "
                "Install with: uv add 'llm-security-pipeline[mysql]'"
            ) from exc
        owns_pool = pool is None
        if pool is None:
            if not aiomysql_connect_kwargs:
                raise ValueError("Provide either connection kwargs (host, user, ...) or an existing pool.")
            pool = await aiomysql.create_pool(**aiomysql_connect_kwargs)
        await ensure_schema(pool)
        return cls(pool, owns_pool=owns_pool)

    async def aclose(self) -> None:
        if self._owns_pool:
            self.pool.close()
            await self.pool.wait_closed()
