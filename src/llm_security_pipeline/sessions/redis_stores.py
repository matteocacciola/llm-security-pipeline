"""
redis_stores.py
Redis implementations of the store interfaces defined in stores.py, for
multi-process/multi-pod deployments.

Cluster model: session keys are wrapped in a Redis Cluster hash tag
(`sentinel:session:{<session_id>}:risk`) so that every key a script or a
reset touches for a given session resolves to the same hash slot. The
topology is detected on first use, so the same code is correct on a
single node and on a cluster with no configuration; see redis_keys.py.

Atomicity model: every check-and-update sequence runs as a single Lua
script. Redis executes each script atomically (single-threaded,
run-to-completion), so "read current count, compare, increment, set TTL"
cannot interleave across processes the way separate GET/SET calls would.
Python-level locking could never provide this: locks don't reach across
OS processes, let alone across pods.

Requires the optional `redis` extra:
    uv add "llm-security-pipeline[redis]"
"""

from __future__ import annotations

import secrets
import time

from .redis_keys import HashTagPolicy, hash_tag, validate_key_prefix
from .stores import NonceStore, ProvenanceRecord, ProvenanceStore, RiskUpdate, SessionStore

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover - redis is an optional dependency
    Redis = None  # type: ignore


# Atomic "increment with TTL set on first write". Used by the nonce store
# for token use counts, where a fixed TTL is the right shape: a token's
# uses are bounded over its lifetime, not over a trailing window. (The
# session budgets used to share it; they moved to the sliding-window
# script below.) One Lua script rather than INCR then EXPIRE, so a crash
# between the two cannot leave a key with no TTL.
_INCR_WITH_TTL_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if tonumber(current) == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""

# Atomic time-decayed risk accumulator. Using a continuous decay (rather
# than resetting on fixed window boundaries) avoids the "reset exactly
# before the attacker's last message" edge case and behaves consistently
# regardless of which process/pod handles which request.
_ADD_RISK_SCRIPT = """
local risk_key = KEYS[1]
local last_key = KEYS[2]
local flag_key = KEYS[3]
local risk_delta = tonumber(ARGV[1])
local decay_per_second = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local flag_threshold = tonumber(ARGV[5])

local current = tonumber(redis.call('GET', risk_key) or '0')
local last = tonumber(redis.call('GET', last_key) or now)
local elapsed = math.max(0, now - last)
local decayed = math.max(0, current - elapsed * decay_per_second)
local updated = decayed + risk_delta

redis.call('SET', risk_key, updated, 'EX', ttl)
redis.call('SET', last_key, now, 'EX', ttl)

if updated >= flag_threshold then
    redis.call('SET', flag_key, '1', 'EX', ttl)
end

-- Both values in one reply: the caller would otherwise come straight back
-- with EXISTS on the flag key to learn what this script just decided.
return {tostring(updated), redis.call('EXISTS', flag_key)}
"""

# Sliding window over a sorted set of event timestamps. ZREMRANGEBYSCORE
# drops what has aged out of the window, ZADD records this event under a
# member unique even at identical timestamps, ZCARD is the answer. Atomic
# because it is one script, so two processes cannot both read N.
_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local member = ARGV[3]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, math.ceil(window))
return redis.call('ZCARD', key)
"""


def _require_redis() -> None:
    if Redis is None:
        raise RuntimeError(
            "The 'redis' package is required for Redis-backed stores. "
            "Install with: uv add 'llm-security-pipeline[redis]'"
        )


class RedisNonceStore(NonceStore):
    """Nonce keys are deliberately left untagged.

    The increment script touches exactly one key, so it is already
    cluster-safe, and untagged nonces spread evenly across slots. Adding a
    tag purely for symmetry with the session store would rename every key
    on upgrade, which for an anti-replay counter means previously spent
    nonces read as unused until the old keys expire — a brief replay window
    bought for nothing.
    """

    def __init__(self, redis_client: "Redis", key_prefix: str = "sentinel:nonce:"):
        _require_redis()
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._script = self._redis.register_script(_INCR_WITH_TTL_SCRIPT)

    async def check_and_increment(self, nonce: str, max_uses: int, ttl_seconds: int) -> int:
        key = f"{self._key_prefix}{nonce}"
        result = await self._script(keys=[key], args=[max(ttl_seconds, 1)])
        return int(result)


# Every field the store keeps for one session, in deletion order:
# `flagged` goes last so that an interrupted reset leaves the session
# flagged rather than cleared, i.e. fails closed.
_SESSION_FIELDS = ("requests", "tool_calls", "risk", "risk_last", "flagged")


class RedisSessionStore(SessionStore):
    """Session counters and cumulative risk, cluster-safe by construction.

    `hash_tags` selects the key layout: "auto" (default) detects Redis
    Cluster on first use, True always tags, False never tags and raises if
    the client turns out to be a cluster client. See redis_keys.py.
    """

    def __init__(
        self,
        redis_client: "Redis",
        key_prefix: str = "sentinel:session:",
        hash_tags: bool | str = "auto",
    ):
        _require_redis()
        validate_key_prefix(key_prefix)
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._tags = HashTagPolicy(hash_tags)
        self._window_script = self._redis.register_script(_SLIDING_WINDOW_SCRIPT)
        self._add_risk_script = self._redis.register_script(_ADD_RISK_SCRIPT)

    async def _keys(self, session_id: str, *fields: str) -> list[str]:
        """Build keys for `session_id`, all guaranteed to share a hash slot
        when tagging is in effect."""
        base = f"{self._key_prefix}{session_id}"
        if await self._tags.enabled(self._redis):
            base = f"{self._key_prefix}{{{hash_tag(session_id)}}}"

        return [f"{base}:{field}" for field in fields]

    async def _slide(self, session_id: str, field: str, window_seconds: int) -> int:
        (key,) = await self._keys(session_id, field)
        result = await self._window_script(
            keys=[key], args=[time.time(), window_seconds, secrets.token_hex(8)],
        )
        return int(result)

    async def increment_requests(self, session_id: str, window_seconds: int) -> int:
        return await self._slide(session_id, "requests", window_seconds)

    async def increment_tool_calls(self, session_id: str, window_seconds: int) -> int:
        return await self._slide(session_id, "tool_calls", window_seconds)

    async def add_risk(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> RiskUpdate:
        keys = await self._keys(session_id, "risk", "risk_last", "flagged")
        updated, flagged = await self._add_risk_script(
            keys=keys,
            args=[risk_delta, decay_per_second, time.time(), ttl_seconds, flag_threshold],
        )
        return RiskUpdate(float(updated), bool(int(flagged)))

    async def is_flagged(self, session_id: str) -> bool:
        (flag_key,) = await self._keys(session_id, "flagged")
        return bool(await self._redis.exists(flag_key))

    async def reset_session(self, session_id: str) -> None:
        keys = await self._keys(session_id, *_SESSION_FIELDS)
        if self._tags.resolved:
            # One slot, so one round trip.
            await self._redis.delete(*keys)
            return
        # Untagged layout: a multi-key DEL is only safe on a single node,
        # and this branch cannot assume one. Deleting individually costs a
        # round trip per field and gives up atomicity, which reset does not
        # need — the field order makes a partial reset fail closed.
        for key in keys:
            await self._redis.delete(key)


class RedisProvenanceStore(ProvenanceStore):
    """Ingest verdicts as Redis hashes, one key per document.

    Single-key operations throughout, so this is cluster-safe without hash
    tags. No TTL is set: a provenance record has to outlive whatever
    session happened to retrieve the document, and an expired record is
    indistinguishable from a document that was never scanned.
    """

    def __init__(self, redis_client: "Redis", key_prefix: str = "sentinel:provenance:"):
        _require_redis()
        self._redis = redis_client
        self._key_prefix = key_prefix

    def _key(self, document_id: str) -> str:
        return f"{self._key_prefix}{document_id}"

    async def record(self, record: ProvenanceRecord) -> None:
        # to_dict() is dict[str, str] throughout; the redis-py stubs spell
        # the accepted mapping type more narrowly than the server does.
        await self._redis.hset(
            self._key(record.document_id),
            mapping=record.to_dict(),  # type: ignore[arg-type]
        )

    async def get(self, document_id: str) -> ProvenanceRecord | None:
        data = await self._redis.hgetall(self._key(document_id))
        if not data:
            return None
        return ProvenanceRecord.from_dict(data)

    async def delete(self, document_id: str) -> None:
        await self._redis.delete(self._key(document_id))
