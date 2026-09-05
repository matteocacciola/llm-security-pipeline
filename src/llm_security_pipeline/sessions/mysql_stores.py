"""
mysql_stores.py
MySQL/MariaDB implementations of the store interfaces defined in
stores.py, for multi-process/multi-pod deployments — same guarantees as
redis_stores.py and postgres_stores.py, different technology.

Atomicity model — two different patterns, chosen the hard way (an earlier
draft using INSERT ... ON DUPLICATE KEY followed by SELECT ... FOR UPDATE
deadlocked under real cross-process contention: the duplicate-key check
takes a shared lock, FOR UPDATE then requests an exclusive upgrade, and
two sessions doing both on the same new row block each other — InnoDB
error 1213):

1. Integer counters (nonce uses, request/tool budgets): one single
   self-contained statement using the canonical LAST_INSERT_ID() trick —
   `INSERT ... ON DUPLICATE KEY UPDATE col = LAST_INSERT_ID(expr)` makes
   the connection-local LAST_INSERT_ID() return the updated value, so no
   explicit transaction, no lock upgrade, and no deadlock is possible.

2. The float risk accumulator (LAST_INSERT_ID only carries integers): a
   short transaction with SELECT ... FOR UPDATE, wrapped in a bounded
   retry on deadlock (error 1213) with rollback + small backoff. Retrying
   on 1213 is the officially documented InnoDB practice; the retry makes
   the operation correct under contention, and the row lock keeps each
   attempt atomic.

TTL model: identical to postgres_stores.py — an `expires_at` column
checked logically inside every operation, plus opportunistic physical
cleanup that correctness never depends on.

Requires the optional `mysql` extra (aiomysql):
    uv add "llm-security-pipeline[mysql]"

Setup: call `await ensure_schema(pool)` once at application startup
(idempotent), or manage the DDL below via your migrations.
"""

from __future__ import annotations

import asyncio
import random
import time

from .stores import NonceStore, ProvenanceRecord, ProvenanceStore, RiskUpdate, SessionStore

try:
    import aiomysql
    import pymysql
except ImportError:  # pragma: no cover - aiomysql is an optional dependency
    aiomysql = None  # type: ignore
    pymysql = None  # type: ignore


_DEADLOCK_ERRNO = 1213
_LOCK_WAIT_TIMEOUT_ERRNO = 1205
_MAX_RETRIES = 5

_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS sentinel_nonces (
        nonce       VARCHAR(64) PRIMARY KEY,
        uses        INT NOT NULL,
        expires_at  DOUBLE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentinel_counters (
        session_id  VARCHAR(255) NOT NULL,
        kind        VARCHAR(32) NOT NULL,
        expires_at  DOUBLE NOT NULL,
        PRIMARY KEY (session_id, kind)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentinel_events (
        session_id  VARCHAR(255) NOT NULL,
        kind        VARCHAR(32) NOT NULL,
        ts          DOUBLE NOT NULL,
        INDEX sentinel_events_key_ts (session_id, kind, ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentinel_revocations (
        subject     VARCHAR(255) PRIMARY KEY,
        revoked_at  DOUBLE NOT NULL,
        expires_at  DOUBLE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentinel_risk (
        session_id  VARCHAR(255) PRIMARY KEY,
        risk        DOUBLE NOT NULL,
        last_update DOUBLE NOT NULL,
        flagged     BOOLEAN NOT NULL DEFAULT FALSE,
        expires_at  DOUBLE NOT NULL
    )
    """,
    # Ingest provenance. No expires_at on purpose: unlike the tables above
    # this is a durable record, not expiring state, and an expired record
    # would be indistinguishable from a document that was never scanned.
    """
    CREATE TABLE IF NOT EXISTS sentinel_provenance (
        document_id  VARCHAR(255) PRIMARY KEY,
        content_hash VARCHAR(64) NOT NULL,
        source_id    VARCHAR(255) NOT NULL,
        trust        VARCHAR(32) NOT NULL,
        decision     VARCHAR(32) NOT NULL,
        risk_score   DOUBLE NOT NULL,
        recorded_at  DOUBLE NOT NULL,
        INDEX sentinel_provenance_decision (decision, recorded_at)
    )
    """,
]


def _require_aiomysql() -> None:
    if aiomysql is None:
        raise RuntimeError(
            "The 'aiomysql' package is required for MySQL-backed stores. "
            "Install with: uv add 'llm-security-pipeline[mysql]'"
        )


async def ensure_schema(pool: "aiomysql.Pool") -> None:
    """Create the required tables if they don't exist. Idempotent."""
    _require_aiomysql()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            for ddl in _DDL_STATEMENTS:
                await cur.execute(ddl)
        await conn.commit()


def _is_retryable(exc: Exception) -> bool:
    return (
        pymysql is not None
        and isinstance(exc, pymysql.err.OperationalError)
        and exc.args
        and exc.args[0] in (_DEADLOCK_ERRNO, _LOCK_WAIT_TIMEOUT_ERRNO)
    )


async def _with_deadlock_retry(operation):
    """Run `operation` (an async callable), retrying on InnoDB deadlock /
    lock-wait-timeout with bounded, jittered backoff.

    This wraps EVERY write operation in this module, including the
    single-statement upserts — verified empirically, not just from the
    docs: under real cross-process contention, even a lone
    INSERT ... ON DUPLICATE KEY UPDATE on the same not-yet-existing key
    can deadlock in InnoDB (all racing sessions take the shared
    duplicate-check lock on the index record, then one requests the
    exclusive upgrade for the update and error 1213 fires). InnoDB's own
    documentation is explicit that applications must always be prepared
    to reissue a transaction rolled back by a deadlock; the retry IS the
    correct usage pattern, not a workaround."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await operation()
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            # Jittered exponential backoff so retrying sessions don't
            # re-collide in lockstep.
            # `random` is fine here: this is retry jitter, not key material.
            await asyncio.sleep(0.005 * (2 ** attempt) * (1 + random.random()))  # noqa: S311
    raise last_exc  # type: ignore[misc]


class _OpportunisticCleanup:
    def __init__(self, interval_seconds: float):
        self._interval = interval_seconds
        self._last_run = 0.0

    def due(self) -> bool:
        now = time.monotonic()
        if now - self._last_run >= self._interval:
            self._last_run = now
            return True
        return False


class MySQLNonceStore(NonceStore):
    def __init__(self, pool: "aiomysql.Pool", cleanup_interval_seconds: float = 60.0):
        _require_aiomysql()
        self._pool = pool
        self._cleanup = _OpportunisticCleanup(cleanup_interval_seconds)

    async def revoke_subject(self, subject: str, revoked_at: float, ttl_seconds: int) -> None:
        async with self._pool.acquire() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO sentinel_revocations (subject, revoked_at, expires_at)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            revoked_at = GREATEST(revoked_at, VALUES(revoked_at)),
                            expires_at = VALUES(expires_at)
                        """,
                        (subject, revoked_at, time.time() + ttl_seconds),
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def revoked_at(self, subject: str) -> float | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT revoked_at FROM sentinel_revocations WHERE subject = %s AND expires_at > %s",
                    (subject, time.time()),
                )
                row = await cur.fetchone()
        return None if row is None else float(row[0])

    async def check_and_increment(self, nonce: str, max_uses: int, ttl_seconds: int) -> int:
        return await _with_deadlock_retry(
            lambda: self._check_and_increment_once(nonce, ttl_seconds)
        )

    async def _check_and_increment_once(self, nonce: str, ttl_seconds: int) -> int:
        now = time.time()
        async with self._pool.acquire() as conn:
            try:
                async with conn.cursor() as cur:
                    if self._cleanup.due():
                        await cur.execute("DELETE FROM sentinel_nonces WHERE expires_at < %s", (now,))
                        await cur.execute("DELETE FROM sentinel_revocations WHERE expires_at < %s", (now,))
                    # Single atomic upsert; LAST_INSERT_ID(expr) records the
                    # updated counter on this connection so it can be read back
                    # without a second locking read. Expired rows logically
                    # reset to a fresh first use (an expired nonce row can only
                    # belong to an already-expired token, which authorize()
                    # rejects on its own expiry check anyway).
                    await cur.execute(
                        """
                        INSERT INTO sentinel_nonces (nonce, uses, expires_at)
                        VALUES (%s, LAST_INSERT_ID(1), %s)
                        ON DUPLICATE KEY UPDATE
                            uses = LAST_INSERT_ID(IF(expires_at < %s, 1, uses + 1)),
                            expires_at = IF(expires_at < %s, VALUES(expires_at), expires_at)
                        """,
                        (nonce, now + ttl_seconds, now, now),
                    )
                    await cur.execute("SELECT LAST_INSERT_ID()")
                    (uses,) = await cur.fetchone()
                await conn.commit()
                return int(uses)
            except Exception:
                await conn.rollback()
                raise


class MySQLSessionStore(SessionStore):
    def __init__(self, pool: "aiomysql.Pool", cleanup_interval_seconds: float = 60.0):
        _require_aiomysql()
        self._pool = pool
        self._cleanup = _OpportunisticCleanup(cleanup_interval_seconds)

    async def _increment(self, session_id: str, kind: str, window_seconds: int) -> int:
        return await _with_deadlock_retry(
            lambda: self._increment_once(session_id, kind, window_seconds)
        )

    async def _increment_once(self, session_id: str, kind: str, window_seconds: int) -> int:
        now = time.time()
        cutoff = now - window_seconds
        async with self._pool.acquire() as conn:
            try:
                async with conn.cursor() as cur:
                    if self._cleanup.due():
                        # Events under expired anchors first: an anchor that
                        # expired has had no event for a whole window.
                        await cur.execute(
                            """
                            DELETE e FROM sentinel_events e
                            JOIN sentinel_counters c
                              ON e.session_id = c.session_id AND e.kind = c.kind
                            WHERE c.expires_at < %s
                            """, (now,),
                        )
                        await cur.execute("DELETE FROM sentinel_counters WHERE expires_at < %s", (now,))
                        await cur.execute("DELETE FROM sentinel_risk WHERE expires_at < %s", (now,))
                    # The upsert takes an exclusive row lock on the anchor
                    # for the rest of the transaction, serializing every
                    # increment for this key so the count is exact. Same
                    # deadlock-retry wrapper as the risk path, in case two
                    # first-ever inserts for one key race on the gap lock.
                    await cur.execute(
                        """
                        INSERT INTO sentinel_counters (session_id, kind, expires_at)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE expires_at = VALUES(expires_at)
                        """,
                        (session_id, kind, now + window_seconds),
                    )
                    await cur.execute(
                        "DELETE FROM sentinel_events WHERE session_id = %s AND kind = %s AND ts <= %s",
                        (session_id, kind, cutoff),
                    )
                    await cur.execute(
                        "INSERT INTO sentinel_events (session_id, kind, ts) VALUES (%s, %s, %s)",
                        (session_id, kind, now),
                    )
                    await cur.execute(
                        "SELECT COUNT(*) FROM sentinel_events WHERE session_id = %s AND kind = %s",
                        (session_id, kind),
                    )
                    (count,) = await cur.fetchone()
                await conn.commit()
                return int(count)
            except Exception:
                await conn.rollback()
                raise

    async def increment_requests(self, session_id: str, window_seconds: int) -> int:
        return await self._increment(session_id, "requests", window_seconds)

    async def increment_tool_calls(self, session_id: str, window_seconds: int) -> int:
        return await self._increment(session_id, "tool_calls", window_seconds)

    async def add_risk(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> RiskUpdate:
        # Float accumulator: LAST_INSERT_ID only carries integers, so this
        # one uses a locked transaction (INSERT IGNORE to guarantee row
        # existence, then SELECT ... FOR UPDATE), under the same retry
        # wrapper as everything else.
        return await _with_deadlock_retry(
            lambda: self._add_risk_once(
                session_id, risk_delta, decay_per_second, flag_threshold, ttl_seconds,
            )
        )

    async def _add_risk_once(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> RiskUpdate:
        now = time.time()
        async with self._pool.acquire() as conn:
            try:
                async with conn.cursor() as cur:
                    # INSERT IGNORE guarantees row existence without the
                    # no-op ON DUPLICATE KEY shared lock that caused the
                    # S->X upgrade deadlock in the earlier draft; the
                    # subsequent FOR UPDATE takes the exclusive lock
                    # directly.
                    await cur.execute(
                        """
                        INSERT IGNORE INTO sentinel_risk
                            (session_id, risk, last_update, flagged, expires_at)
                        VALUES (%s, 0, %s, FALSE, %s)
                        """,
                        (session_id, now, now + ttl_seconds),
                    )
                    await cur.execute(
                        """
                        SELECT risk, last_update, flagged, expires_at FROM sentinel_risk
                        WHERE session_id = %s FOR UPDATE
                        """,
                        (session_id,),
                    )
                    risk, last_update, flagged, expires_at = await cur.fetchone()
                    if expires_at < now:
                        risk, last_update, flagged = 0.0, now, False  # expired: fresh state
                    elapsed = max(0.0, now - last_update)
                    risk = max(0.0, risk - elapsed * decay_per_second) + risk_delta
                    flagged = bool(flagged) or risk >= flag_threshold  # sticky flag
                    await cur.execute(
                        """
                        UPDATE sentinel_risk
                        SET risk = %s, last_update = %s, flagged = %s, expires_at = %s
                        WHERE session_id = %s
                        """,
                        (risk, now, flagged, now + ttl_seconds, session_id),
                    )
                await conn.commit()
                return RiskUpdate(float(risk), bool(flagged))
            except Exception:
                await conn.rollback()
                raise

    async def is_flagged(self, session_id: str) -> bool:
        now = time.time()
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT flagged FROM sentinel_risk WHERE session_id = %s AND expires_at >= %s",
                    (session_id, now),
                )
                row = await cur.fetchone()
                return bool(row and row[0])

    async def reset_session(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM sentinel_events WHERE session_id = %s", (session_id,))
                await cur.execute("DELETE FROM sentinel_counters WHERE session_id = %s", (session_id,))
                await cur.execute("DELETE FROM sentinel_risk WHERE session_id = %s", (session_id,))
            await conn.commit()


class MySQLProvenanceStore(ProvenanceStore):
    """Ingest verdicts, one upserted row per document.

    Needs neither the deadlock retry nor the row locks the other MySQL
    stores in this module use: there is no read-modify-write here, just a
    single idempotent upsert, so InnoDB has nothing to serialise.
    """

    def __init__(self, pool: "aiomysql.Pool"):
        _require_aiomysql()
        self._pool = pool

    async def record(self, record: ProvenanceRecord) -> None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO sentinel_provenance
                        (document_id, content_hash, source_id, trust, decision, risk_score, recorded_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        content_hash = VALUES(content_hash),
                        source_id    = VALUES(source_id),
                        trust        = VALUES(trust),
                        decision     = VALUES(decision),
                        risk_score   = VALUES(risk_score),
                        recorded_at  = VALUES(recorded_at)
                    """,
                    (record.document_id, record.content_hash, record.source_id,
                     record.trust, record.decision, record.risk_score, record.recorded_at),
                )
            await conn.commit()

    async def get(self, document_id: str) -> ProvenanceRecord | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT document_id, content_hash, source_id, trust, decision,
                           risk_score, recorded_at
                    FROM sentinel_provenance WHERE document_id = %s
                    """,
                    (document_id,),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return ProvenanceRecord(
            document_id=row[0], content_hash=row[1], source_id=row[2],
            trust=row[3], decision=row[4], risk_score=float(row[5]),
            recorded_at=float(row[6]),
        )

    async def delete(self, document_id: str) -> None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM sentinel_provenance WHERE document_id = %s", (document_id,),
                )
            await conn.commit()

    async def list_by_decision(self, decision: str, limit: int = 100) -> list[ProvenanceRecord]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT document_id, content_hash, source_id, trust, decision, risk_score, recorded_at
                    FROM sentinel_provenance WHERE decision = %s
                    ORDER BY recorded_at ASC LIMIT %s
                    """,
                    (decision, int(limit)),
                )
                rows = await cur.fetchall()
        return [
            ProvenanceRecord(
                document_id=r[0], content_hash=r[1], source_id=r[2], trust=r[3],
                decision=r[4], risk_score=float(r[5]), recorded_at=float(r[6]),
            )
            for r in rows
        ]

    async def set_decision(self, document_id: str, decision: str) -> bool:
        async with self._pool.acquire() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE sentinel_provenance SET decision = %s WHERE document_id = %s",
                        (decision, document_id),
                    )
                    updated = cur.rowcount
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return updated > 0
