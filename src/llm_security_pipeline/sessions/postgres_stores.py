"""
postgres_stores.py
PostgreSQL implementations of the store interfaces defined in stores.py,
for multi-process/multi-pod deployments — same guarantees as
redis_stores.py, different technology.

Atomicity model: every check-and-update sequence is a SINGLE SQL statement
(`INSERT ... ON CONFLICT ... DO UPDATE ... RETURNING`), which PostgreSQL
executes atomically with respect to concurrent statements on the same row
(row-level locking). This is the SQL equivalent of the Lua scripts in
redis_stores.py: no separate read-then-write window in which another
process could interleave.

TTL model: Redis expires keys natively; SQL does not. Expiry is emulated
with an `expires_at` column that is (a) checked inside the atomic
statement itself, so an expired row behaves as absent/reset regardless of
whether it was physically deleted, and (b) physically cleaned up
opportunistically (a cheap DELETE of expired rows, throttled to at most
once per `cleanup_interval_seconds` per process) so tables don't grow
unboundedly. Correctness never depends on the cleanup having run.

Requires the optional `postgres` extra (asyncpg):
    uv add "llm-security-pipeline[postgres]"

Setup: call `await ensure_schema(pool)` once at application startup (it is
idempotent), or create the tables yourself from the DDL below if schema
changes are managed by migrations.
"""

from __future__ import annotations

import time

from .stores import NonceStore, SessionStore

try:
    import asyncpg
except ImportError:  # pragma: no cover - asyncpg is an optional dependency
    asyncpg = None  # type: ignore


_DDL = """
CREATE TABLE IF NOT EXISTS sentinel_nonces (
    nonce       TEXT PRIMARY KEY,
    uses        INTEGER NOT NULL,
    expires_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS sentinel_counters (
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    count       INTEGER NOT NULL,
    expires_at  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (session_id, kind)
);

CREATE TABLE IF NOT EXISTS sentinel_risk (
    session_id  TEXT PRIMARY KEY,
    risk        DOUBLE PRECISION NOT NULL,
    last_update DOUBLE PRECISION NOT NULL,
    flagged     BOOLEAN NOT NULL DEFAULT FALSE,
    expires_at  DOUBLE PRECISION NOT NULL
);
"""


def _require_asyncpg() -> None:
    if asyncpg is None:
        raise RuntimeError(
            "The 'asyncpg' package is required for Postgres-backed stores. "
            "Install with: uv add 'llm-security-pipeline[postgres]'"
        )


async def ensure_schema(pool: "asyncpg.Pool") -> None:
    """Create the required tables if they don't exist. Idempotent; call
    once at application startup, or manage the DDL via your migrations."""
    _require_asyncpg()
    async with pool.acquire() as conn:
        await conn.execute(_DDL)


class _OpportunisticCleanup:
    """Throttles physical deletion of expired rows to at most once per
    interval per process. Correctness never depends on it: expiry is also
    enforced logically inside every atomic statement."""

    def __init__(self, interval_seconds: float):
        self._interval = interval_seconds
        self._last_run = 0.0

    def due(self) -> bool:
        now = time.monotonic()
        if now - self._last_run >= self._interval:
            self._last_run = now
            return True
        return False


class PostgresNonceStore(NonceStore):
    def __init__(self, pool: "asyncpg.Pool", cleanup_interval_seconds: float = 60.0):
        _require_asyncpg()
        self._pool = pool
        self._cleanup = _OpportunisticCleanup(cleanup_interval_seconds)

    async def check_and_increment(self, nonce: str, max_uses: int, ttl_seconds: int) -> int:
        now = time.time()
        async with self._pool.acquire() as conn:
            if self._cleanup.due():
                await conn.execute("DELETE FROM sentinel_nonces WHERE expires_at < $1", now)
            # Single atomic statement: insert first use, or increment.
            # An expired row is logically reset to a fresh first use — the
            # nonce's own TTL matches the token's, so an expired row can
            # only belong to an already-expired token, which authorize()
            # rejects on the expiry check anyway.
            row = await conn.fetchrow(
                """
                INSERT INTO sentinel_nonces (nonce, uses, expires_at)
                VALUES ($1, 1, $2)
                ON CONFLICT (nonce) DO UPDATE SET
                    uses = CASE WHEN sentinel_nonces.expires_at < $3
                                THEN 1 ELSE sentinel_nonces.uses + 1 END,
                    expires_at = CASE WHEN sentinel_nonces.expires_at < $3
                                      THEN $2 ELSE sentinel_nonces.expires_at END
                RETURNING uses
                """,
                nonce, now + ttl_seconds, now,
            )
            return int(row["uses"])


class PostgresSessionStore(SessionStore):
    def __init__(self, pool: "asyncpg.Pool", cleanup_interval_seconds: float = 60.0):
        _require_asyncpg()
        self._pool = pool
        self._cleanup = _OpportunisticCleanup(cleanup_interval_seconds)

    async def _increment(self, session_id: str, kind: str, window_seconds: int) -> int:
        now = time.time()
        async with self._pool.acquire() as conn:
            if self._cleanup.due():
                await conn.execute("DELETE FROM sentinel_counters WHERE expires_at < $1", now)
                await conn.execute("DELETE FROM sentinel_risk WHERE expires_at < $1", now)
            # Fixed-window counter in one atomic statement: an expired
            # window resets to 1, an active one increments.
            row = await conn.fetchrow(
                """
                INSERT INTO sentinel_counters (session_id, kind, count, expires_at)
                VALUES ($1, $2, 1, $3)
                ON CONFLICT (session_id, kind) DO UPDATE SET
                    count = CASE WHEN sentinel_counters.expires_at < $4
                                 THEN 1 ELSE sentinel_counters.count + 1 END,
                    expires_at = CASE WHEN sentinel_counters.expires_at < $4
                                      THEN $3 ELSE sentinel_counters.expires_at END
                RETURNING count
                """,
                session_id, kind, now + window_seconds, now,
            )
            return int(row["count"])

    async def increment_requests(self, session_id: str, window_seconds: int) -> int:
        return await self._increment(session_id, "requests", window_seconds)

    async def increment_tool_calls(self, session_id: str, window_seconds: int) -> int:
        return await self._increment(session_id, "tool_calls", window_seconds)

    async def add_risk(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> float:
        now = time.time()
        async with self._pool.acquire() as conn:
            # Continuous-decay accumulator in one atomic statement — the
            # SQL translation of _ADD_RISK_SCRIPT in redis_stores.py.
            # The flag is sticky: once TRUE it stays TRUE until reset.
            row = await conn.fetchrow(
                """
                INSERT INTO sentinel_risk (session_id, risk, last_update, flagged, expires_at)
                VALUES ($1, $2::float8, $3::float8, $2::float8 >= $4::float8, $5::float8)
                ON CONFLICT (session_id) DO UPDATE SET
                    risk = GREATEST(0, sentinel_risk.risk
                                       - GREATEST(0, $3::float8 - sentinel_risk.last_update) * $6::float8)
                           + $2::float8,
                    last_update = $3::float8,
                    flagged = sentinel_risk.flagged OR
                              (GREATEST(0, sentinel_risk.risk
                                           - GREATEST(0, $3::float8 - sentinel_risk.last_update) * $6::float8)
                               + $2::float8) >= $4::float8,
                    expires_at = $5::float8
                RETURNING risk
                """,
                session_id, risk_delta, now, flag_threshold, now + ttl_seconds, decay_per_second,
            )
            return float(row["risk"])

    async def is_flagged(self, session_id: str) -> bool:
        now = time.time()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT flagged FROM sentinel_risk WHERE session_id = $1 AND expires_at >= $2",
                session_id, now,
            )
            return bool(row and row["flagged"])

    async def reset_session(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM sentinel_counters WHERE session_id = $1", session_id)
            await conn.execute("DELETE FROM sentinel_risk WHERE session_id = $1", session_id)
