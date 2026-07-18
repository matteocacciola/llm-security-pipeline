"""
stores.py
Backend-agnostic interfaces for the pipeline's shared mutable state, plus
in-memory implementations for local development, unit tests and
single-process deployments.

Why these interfaces exist: the pipeline's cross-process guarantees
(token anti-replay, session rate limits, cumulative risk) require a
centralized store with ATOMIC check-and-update semantics. That is a
property, not a product — Redis implements it (see redis_stores.py), and
so can PostgreSQL (transactions / INSERT ... ON CONFLICT), MySQL
(INSERT ... ON DUPLICATE KEY UPDATE), DynamoDB (conditional writes), or
anything comparable. Implement these two small interfaces on your
technology of choice and bundle them in a StateBackend (state_backend.py).

The in-memory implementations below are NOT safe across multiple
processes/pods: each process would keep its own private counters, and
every limit silently becomes per-process instead of global. Use them only
where a single Python process handles all traffic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Nonce / capability-token usage store (backs scope_guard.py)
# ---------------------------------------------------------------------------

class NonceStore(ABC):
    """Tracks how many times a capability token's nonce has been consumed,
    atomically, so max_uses / anti-replay holds across processes."""

    @abstractmethod
    async def check_and_increment(self, nonce: str, max_uses: int, ttl_seconds: int) -> int:
        """Atomically increment the usage counter for `nonce` and return the
        new count. Callers compare the returned count to `max_uses`
        themselves so the store stays a simple counter primitive."""


class InMemoryNonceStore(NonceStore):
    """Single-process implementation for local development and unit tests."""

    def __init__(self):
        self._counts: dict[str, int] = {}

    async def check_and_increment(self, nonce: str, max_uses: int, ttl_seconds: int) -> int:
        # No real concurrency hazard within a single asyncio event loop
        # thread, since there is no `await` between read and write here.
        self._counts[nonce] = self._counts.get(nonce, 0) + 1
        return self._counts[nonce]


# ---------------------------------------------------------------------------
# Session store (backs rate_limiter.py): request/tool-call budget +
# cumulative cross-turn risk with time-based decay.
# ---------------------------------------------------------------------------

@dataclass
class SessionCounters:
    request_count: int
    tool_call_count: int
    cumulative_risk: float
    flagged: bool


class SessionStore(ABC):
    @abstractmethod
    async def increment_requests(self, session_id: str, window_seconds: int) -> int:
        ...

    @abstractmethod
    async def increment_tool_calls(self, session_id: str, window_seconds: int) -> int:
        ...

    @abstractmethod
    async def add_risk(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> float:
        """Atomically apply time-based decay to the session's cumulative
        risk, add `risk_delta`, persist the result, and return the updated
        cumulative value. If it crosses `flag_threshold`, mark the session
        as flagged (sticky until reset_session)."""

    @abstractmethod
    async def is_flagged(self, session_id: str) -> bool:
        ...

    @abstractmethod
    async def reset_session(self, session_id: str) -> None:
        ...


class InMemorySessionStore(SessionStore):
    """Single-process implementation for local development and unit tests."""

    def __init__(self):
        self._requests: dict[str, tuple[int, float]] = {}   # session -> (count, window_expiry)
        self._tool_calls: dict[str, tuple[int, float]] = {}
        self._risk: dict[str, tuple[float, float]] = {}      # session -> (value, last_update_ts)
        self._flagged: set[str] = set()

    def _incr(self, store: dict[str, tuple[int, float]], session_id: str, window_seconds: int) -> int:
        now = time.time()
        count, expiry = store.get(session_id, (0, 0.0))
        if now > expiry:
            count, expiry = 0, now + window_seconds
        count += 1
        store[session_id] = (count, expiry)
        return count

    async def increment_requests(self, session_id: str, window_seconds: int) -> int:
        return self._incr(self._requests, session_id, window_seconds)

    async def increment_tool_calls(self, session_id: str, window_seconds: int) -> int:
        return self._incr(self._tool_calls, session_id, window_seconds)

    async def add_risk(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> float:
        now = time.time()
        current, last_update = self._risk.get(session_id, (0.0, now))
        elapsed = max(0.0, now - last_update)
        decayed = max(0.0, current - elapsed * decay_per_second)
        updated = decayed + risk_delta
        self._risk[session_id] = (updated, now)
        if updated >= flag_threshold:
            self._flagged.add(session_id)
        return updated

    async def is_flagged(self, session_id: str) -> bool:
        return session_id in self._flagged

    async def reset_session(self, session_id: str) -> None:
        self._requests.pop(session_id, None)
        self._tool_calls.pop(session_id, None)
        self._risk.pop(session_id, None)
        self._flagged.discard(session_id)
