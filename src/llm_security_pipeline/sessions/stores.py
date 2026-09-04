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
from typing import NamedTuple
from collections import deque
from dataclasses import dataclass, field


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


class RiskUpdate(NamedTuple):
    """What add_risk hands back: the new cumulative score and whether the
    session is flagged AFTER this update (sticky, so it may have been
    flagged already). Returning both saves the caller a second round trip
    to ask a question the store just answered."""

    cumulative: float
    flagged: bool


class SessionStore(ABC):
    """Per-session counters and risk, shared across processes.

    Request and tool-call budgets are **sliding windows**: `increment_*`
    returns how many events, including this one, fall within the trailing
    `window_seconds`. A fixed window (count, reset at the boundary) lets
    a caller who knows where the boundary is spend a full budget just
    before it and another just after, doubling the limit over a few
    seconds; a sliding window means "N per W" holds over *every* interval
    of length W. The cost is one timestamp per event per key, bounded by
    the budget itself, and it expires with the window.

    Under concurrency the count must be exact — two processes incrementing
    together get N+1 and N+2, never N+1 twice — which is the property the
    cross-process tests assert. Redis gets it from the script being
    atomic; the SQL backends take a row lock on a per-key anchor row.
    """

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
    ) -> RiskUpdate:
        """Atomically apply time-based decay to the session's cumulative
        risk, add `risk_delta`, persist the result, and return the updated
        cumulative value together with the flag state. If the value crosses
        `flag_threshold`, mark the session as flagged (sticky until
        reset_session)."""

    @abstractmethod
    async def is_flagged(self, session_id: str) -> bool:
        ...

    @abstractmethod
    async def reset_session(self, session_id: str) -> None:
        ...


class InMemorySessionStore(SessionStore):
    """Single-process implementation for local development and unit tests."""

    def __init__(self):
        self._requests: dict[str, deque[float]] = {}   # session -> event timestamps
        self._tool_calls: dict[str, deque[float]] = {}
        self._risk: dict[str, tuple[float, float]] = {}      # session -> (value, last_update_ts)
        self._flagged: set[str] = set()

    def _incr(self, store: dict[str, deque[float]], session_id: str, window_seconds: int) -> int:
        now = time.time()
        events = store.setdefault(session_id, deque())
        cutoff = now - window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        events.append(now)
        return len(events)

    async def increment_requests(self, session_id: str, window_seconds: int) -> int:
        return self._incr(self._requests, session_id, window_seconds)

    async def increment_tool_calls(self, session_id: str, window_seconds: int) -> int:
        return self._incr(self._tool_calls, session_id, window_seconds)

    async def add_risk(
        self, session_id: str, risk_delta: float, decay_per_second: float,
        flag_threshold: float, ttl_seconds: int,
    ) -> RiskUpdate:
        now = time.time()
        current, last_update = self._risk.get(session_id, (0.0, now))
        elapsed = max(0.0, now - last_update)
        decayed = max(0.0, current - elapsed * decay_per_second)
        updated = decayed + risk_delta
        self._risk[session_id] = (updated, now)
        if updated >= flag_threshold:
            self._flagged.add(session_id)
        return RiskUpdate(updated, session_id in self._flagged)

    async def is_flagged(self, session_id: str) -> bool:
        return session_id in self._flagged

    async def reset_session(self, session_id: str) -> None:
        self._requests.pop(session_id, None)
        self._tool_calls.pop(session_id, None)
        self._risk.pop(session_id, None)
        self._flagged.discard(session_id)


# ---------------------------------------------------------------------------
# Provenance store (backs ingest_guard.py): what was decided about a
# document when it entered the index, and what its bytes were at the time.
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceRecord:
    document_id: str
    content_hash: str
    source_id: str
    trust: str
    decision: str
    risk_score: float
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "source_id": self.source_id,
            "trust": self.trust,
            "decision": self.decision,
            "risk_score": str(self.risk_score),
            "recorded_at": str(self.recorded_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProvenanceRecord":
        def _text(value) -> str:
            return value.decode() if isinstance(value, bytes) else str(value)

        plain = {_text(k): _text(v) for k, v in data.items()}
        return cls(
            document_id=plain["document_id"],
            content_hash=plain["content_hash"],
            source_id=plain["source_id"],
            trust=plain["trust"],
            decision=plain["decision"],
            risk_score=float(plain["risk_score"]),
            recorded_at=float(plain["recorded_at"]),
        )


class ProvenanceStore(ABC):
    """Records the ingest-time verdict for a document so retrieval can
    check that what came back is what was approved.

    Unlike the counters in the other stores this is not ephemeral: a record
    has to outlive the document it describes, so implementations should not
    put a short TTL on it. There is no in-memory-is-fine caveat here for a
    different reason than usual — the risk is not that limits become
    per-process, it is that a restart silently turns every verified
    retrieval into an unverified one.
    """

    @abstractmethod
    async def record(self, record: ProvenanceRecord) -> None:
        ...

    @abstractmethod
    async def get(self, document_id: str) -> ProvenanceRecord | None:
        ...

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        ...


class InMemoryProvenanceStore(ProvenanceStore):
    """Single-process implementation for local development and unit tests."""

    def __init__(self):
        self._records: dict[str, ProvenanceRecord] = {}

    async def record(self, record: ProvenanceRecord) -> None:
        self._records[record.document_id] = record

    async def get(self, document_id: str) -> ProvenanceRecord | None:
        return self._records.get(document_id)

    async def delete(self, document_id: str) -> None:
        self._records.pop(document_id, None)
