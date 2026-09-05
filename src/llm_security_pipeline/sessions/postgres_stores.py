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

from .stores import NonceStore, ProvenanceRecord, ProvenanceStore, RiskUpdate, SessionStore

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

-- Per-key anchor row. Its purpose is the row lock: the upsert on it
-- serializes concurrent increments for one key, which is what makes the
-- count below exact rather than racy. expires_at drives cleanup.
CREATE TABLE IF NOT EXISTS sentinel_counters (
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    expires_at  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (session_id, kind)
);

-- One row per event; the sliding window is a count over the trailing
-- window_seconds. Rows older than the window are deleted on the next
-- increment for that key, and wholesale when the anchor expires.
CREATE TABLE IF NOT EXISTS sentinel_events (
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    ts          DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS sentinel_events_key_ts ON sentinel_events (session_id, kind, ts);

CREATE TABLE IF NOT EXISTS sentinel_risk (
    session_id  TEXT PRIMARY KEY,
    risk        DOUBLE PRECISION NOT NULL,
    last_update DOUBLE PRECISION NOT NULL,
    flagged     BOOLEAN NOT NULL DEFAULT FALSE,
    expires_at  DOUBLE PRECISION NOT NULL
);

-- Ingest provenance. Deliberately has no expires_at: unlike every other
-- table here it is not a counter with a TTL. A record has to outlive
-- whatever session retrieved the document, and an expired record is
-- indistinguishable from a document that was never scanned at all.
CREATE TABLE IF NOT EXISTS sentinel_revocations (
    subject     TEXT PRIMARY KEY,
    revoked_at  DOUBLE PRECISION NOT NULL,
    expires_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS sentinel_provenance (
    document_id  TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    trust        TEXT NOT NULL,
    decision     TEXT NOT NULL,
    risk_score   DOUBLE PRECISION NOT NULL,
    recorded_at  DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS sentinel_provenance_decision ON sentinel_provenance (decision, recorded_at);
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
    async def revoke_subject(self, subject: str, revoked_at: float, ttl_seconds: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sentinel_revocations (subject, revoked_at, expires_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (subject) DO UPDATE SET
                    revoked_at = GREATEST(sentinel_revocations.revoked_at, EXCLUDED.revoked_at),
                    expires_at = EXCLUDED.expires_at
                """,
                subject, revoked_at, time.time() + ttl_seconds,
            )

    async def revoked_at(self, subject: str) -> float | None:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT revoked_at FROM sentinel_revocations WHERE subject = $1 AND expires_at > $2",
                subject, time.time(),
            )
        return None if value is None else float(value)

    def __init__(self, pool: "asyncpg.Pool", cleanup_interval_seconds: float = 60.0):
        _require_asyncpg()
        self._pool = pool
        self._cleanup = _OpportunisticCleanup(cleanup_interval_seconds)

    async def check_and_increment(self, nonce: str, max_uses: int, ttl_seconds: int) -> int:
        now = time.time()
        async with self._pool.acquire() as conn:
            if self._cleanup.due():
                await conn.execute("DELETE FROM sentinel_nonces WHERE expires_at < $1", now)
                await conn.execute("DELETE FROM sentinel_revocations WHERE expires_at < $1", now)
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
        cutoff = now - window_seconds
        async with self._pool.acquire() as conn:
            if self._cleanup.due():
                # Events first, joined on expired anchors: an anchor that
                # has expired has had no event for a whole window, so
                # every event under it is out of the window too.
                await conn.execute(
                    """
                    DELETE FROM sentinel_events e USING sentinel_counters c
                    WHERE e.session_id = c.session_id AND e.kind = c.kind AND c.expires_at < $1
                    """, now,
                )
                await conn.execute("DELETE FROM sentinel_counters WHERE expires_at < $1", now)
                await conn.execute("DELETE FROM sentinel_risk WHERE expires_at < $1", now)
            async with conn.transaction():
                # The upsert takes a row lock on the anchor that lasts for
                # the transaction, serializing every increment for this
                # key. Without it two transactions could both count N.
                await conn.execute(
                    """
                    INSERT INTO sentinel_counters (session_id, kind, expires_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (session_id, kind) DO UPDATE SET expires_at = $3
                    """,
                    session_id, kind, now + window_seconds,
                )
                await conn.execute(
                    "DELETE FROM sentinel_events WHERE session_id = $1 AND kind = $2 AND ts <= $3",
                    session_id, kind, cutoff,
                )
                await conn.execute(
                    "INSERT INTO sentinel_events (session_id, kind, ts) VALUES ($1, $2, $3)",
                    session_id, kind, now,
                )
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM sentinel_events WHERE session_id = $1 AND kind = $2",
                    session_id, kind,
                )
            return int(count)

    async def increment_requests(self, session_id: str, window_seconds: int) -> int:
        return await self._increment(session_id, "requests", window_seconds)

    async def increment_tool_calls(self, session_id: str, window_seconds: int) -> int:
        return await self._increment(session_id, "tool_calls", window_seconds)

    async def add_risk(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> RiskUpdate:
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
                RETURNING risk, flagged
                """,
                session_id, risk_delta, now, flag_threshold, now + ttl_seconds, decay_per_second,
            )
            return RiskUpdate(float(row["risk"]), bool(row["flagged"]))

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
            await conn.execute("DELETE FROM sentinel_events WHERE session_id = $1", session_id)
            await conn.execute("DELETE FROM sentinel_counters WHERE session_id = $1", session_id)
            await conn.execute("DELETE FROM sentinel_risk WHERE session_id = $1", session_id)


class PostgresProvenanceStore(ProvenanceStore):
    """Ingest verdicts in a single upserted row per document.

    No TTL and no opportunistic cleanup, unlike the other Postgres stores
    in this module: provenance is a durable record, not expiring state.
    Deleting a row means the next retrieval of that document reports
    `no_provenance_record`, so removal belongs to whatever removes the
    document from your index.
    """

    def __init__(self, pool: "asyncpg.Pool"):
        _require_asyncpg()
        self._pool = pool

    async def record(self, record: ProvenanceRecord) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sentinel_provenance
                    (document_id, content_hash, source_id, trust, decision, risk_score, recorded_at)
                VALUES ($1, $2, $3, $4, $5, $6::float8, $7::float8)
                ON CONFLICT (document_id) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    source_id    = EXCLUDED.source_id,
                    trust        = EXCLUDED.trust,
                    decision     = EXCLUDED.decision,
                    risk_score   = EXCLUDED.risk_score,
                    recorded_at  = EXCLUDED.recorded_at
                """,
                record.document_id, record.content_hash, record.source_id,
                record.trust, record.decision, record.risk_score, record.recorded_at,
            )

    async def get(self, document_id: str) -> ProvenanceRecord | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT document_id, content_hash, source_id, trust, decision,
                       risk_score, recorded_at
                FROM sentinel_provenance WHERE document_id = $1
                """,
                document_id,
            )
        if row is None:
            return None
        return ProvenanceRecord(
            document_id=row["document_id"],
            content_hash=row["content_hash"],
            source_id=row["source_id"],
            trust=row["trust"],
            decision=row["decision"],
            risk_score=float(row["risk_score"]),
            recorded_at=float(row["recorded_at"]),
        )

    async def delete(self, document_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM sentinel_provenance WHERE document_id = $1", document_id)

    async def list_by_decision(self, decision: str, limit: int = 100) -> list[ProvenanceRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT document_id, content_hash, source_id, trust, decision, risk_score, recorded_at
                FROM sentinel_provenance WHERE decision = $1
                ORDER BY recorded_at ASC LIMIT $2
                """,
                decision, limit,
            )
        return [
            ProvenanceRecord(
                document_id=r["document_id"], content_hash=r["content_hash"], source_id=r["source_id"],
                trust=r["trust"], decision=r["decision"], risk_score=float(r["risk_score"]),
                recorded_at=float(r["recorded_at"]),
            )
            for r in rows
        ]

    async def set_decision(self, document_id: str, decision: str) -> bool:
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                "UPDATE sentinel_provenance SET decision = $2 WHERE document_id = $1 RETURNING document_id",
                document_id, decision,
            )
        return updated is not None
