"""
redis_stores.py
Redis implementations of the store interfaces defined in stores.py, for
multi-process/multi-pod deployments.

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

import time

from .stores import NonceStore, SessionStore

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover - redis is an optional dependency
    Redis = None  # type: ignore


# Atomic "increment with TTL set on first write" — the standard safe
# rate-limiting primitive in Redis. Doing this as one Lua script avoids the
# race where a process crashes between INCR and EXPIRE, which would leave
# the key without a TTL and leak memory forever.
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

return tostring(updated)
"""


def _require_redis() -> None:
    if Redis is None:
        raise RuntimeError(
            "The 'redis' package is required for Redis-backed stores. "
            "Install with: uv add 'llm-security-pipeline[redis]'"
        )


class RedisNonceStore(NonceStore):
    def __init__(self, redis_client: "Redis", key_prefix: str = "sentinel:nonce:"):
        _require_redis()
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._script = self._redis.register_script(_INCR_WITH_TTL_SCRIPT)

    async def check_and_increment(self, nonce: str, max_uses: int, ttl_seconds: int) -> int:
        key = f"{self._key_prefix}{nonce}"
        result = await self._script(keys=[key], args=[max(ttl_seconds, 1)])
        return int(result)


class RedisSessionStore(SessionStore):
    def __init__(self, redis_client: "Redis", key_prefix: str = "sentinel:session:"):
        _require_redis()
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._incr_script = self._redis.register_script(_INCR_WITH_TTL_SCRIPT)
        self._add_risk_script = self._redis.register_script(_ADD_RISK_SCRIPT)

    async def increment_requests(self, session_id: str, window_seconds: int) -> int:
        key = f"{self._key_prefix}{session_id}:requests"
        result = await self._incr_script(keys=[key], args=[window_seconds])
        return int(result)

    async def increment_tool_calls(self, session_id: str, window_seconds: int) -> int:
        key = f"{self._key_prefix}{session_id}:tool_calls"
        result = await self._incr_script(keys=[key], args=[window_seconds])
        return int(result)

    async def add_risk(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> float:
        risk_key = f"{self._key_prefix}{session_id}:risk"
        last_key = f"{self._key_prefix}{session_id}:risk_last"
        flag_key = f"{self._key_prefix}{session_id}:flagged"
        result = await self._add_risk_script(
            keys=[risk_key, last_key, flag_key],
            args=[risk_delta, decay_per_second, time.time(), ttl_seconds, flag_threshold],
        )
        return float(result)

    async def is_flagged(self, session_id: str) -> bool:
        flag_key = f"{self._key_prefix}{session_id}:flagged"
        return bool(await self._redis.exists(flag_key))

    async def reset_session(self, session_id: str) -> None:
        keys = [
            f"{self._key_prefix}{session_id}:requests",
            f"{self._key_prefix}{session_id}:tool_calls",
            f"{self._key_prefix}{session_id}:risk",
            f"{self._key_prefix}{session_id}:risk_last",
            f"{self._key_prefix}{session_id}:flagged",
        ]
        await self._redis.delete(*keys)
