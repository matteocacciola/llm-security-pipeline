"""
rate_limiter.py
Mitigates two gaps not covered by input/output scanning alone:

1. Denial-of-service / resource exhaustion: a single request may look
   harmless on its own, but an agent that is tricked or abused into making
   excessive requests or tool calls in a short window can still cause
   damage (cost, cascading calls, rate-limit exhaustion on downstream APIs).
2. Multi-turn jailbreak build-up: an attack assembled gradually across
   several messages (e.g. progressive role-play) can stay under the
   single-turn risk threshold at every step while still succeeding overall.
   Tracking a cumulative risk score per session catches this pattern.

Production note: all counters are delegated to a SessionStore (see
stores.py / redis_stores.py). Use RedisSessionStore in any deployment with more than
one process/pod, otherwise each instance enforces its own private budget
and the effective limit becomes (configured_limit * number_of_instances) —
usually not what you want, and actively dangerous for the DoS-mitigation
use case this module exists for.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..sessions.stores import SessionStore, InMemorySessionStore


class RateLimitExceeded(Exception):
    pass


@dataclass
class SessionLimits:
    """Configurable limits for a single session/time window."""
    window_seconds: int = 60
    max_requests_per_window: int = 30
    max_tool_calls_per_window: int = 15
    # Cumulative risk threshold across turns in a session: catches
    # gradually-escalating jailbreak attempts that stay under the
    # single-turn `input_risk_threshold` at every individual step.
    cumulative_risk_threshold: float = 1.5
    # Continuous time-based decay (risk units per second) rather than a
    # fixed-window reset, so the behavior doesn't depend on which
    # process/pod happens to handle a given request.
    risk_decay_per_second: float = 0.02
    # How long session state survives with no activity before Redis
    # expires it. Should comfortably exceed window_seconds.
    session_ttl_seconds: int = 3600


class SessionRateLimiter:
    """Tracks per-session request/tool-call volume and cumulative risk.

    Typical usage inside a pipeline:
        limiter = SessionRateLimiter(store=RedisSessionStore(redis_client))

        await limiter.check_request(session_id)           # raises RateLimitExceeded if over budget
        await limiter.record_turn_risk(session_id, 0.4)   # feed each turn's sanitizer risk_score
        if await limiter.is_session_flagged(session_id):
            ...                                           # escalate: extra scrutiny, human review, etc.

        await limiter.check_tool_call(session_id)          # before authorizing a tool call
    """

    def __init__(self, limits: SessionLimits | None = None, store: SessionStore | None = None):
        self.limits = limits or SessionLimits()
        # Defaults to an in-memory store, safe only for a single process.
        # Pass a RedisSessionStore explicitly for multi-process/instance deployments.
        self._store = store or InMemorySessionStore()

    async def check_request(self, session_id: str) -> None:
        count = await self._store.increment_requests(session_id, self.limits.window_seconds)
        if count > self.limits.max_requests_per_window:
            raise RateLimitExceeded(
                f"Session '{session_id}' exceeded {self.limits.max_requests_per_window} "
                f"requests within {self.limits.window_seconds}s."
            )

    async def check_tool_call(self, session_id: str) -> None:
        count = await self._store.increment_tool_calls(session_id, self.limits.window_seconds)
        if count > self.limits.max_tool_calls_per_window:
            raise RateLimitExceeded(
                f"Session '{session_id}' exceeded {self.limits.max_tool_calls_per_window} "
                f"tool calls within {self.limits.window_seconds}s."
            )

    async def record_turn_risk(self, session_id: str, risk_score: float) -> float:
        """Feed the per-turn risk score (e.g. from sanitizer.scan_text) into
        the session's cumulative total. Returns the updated cumulative
        score. Flags the session once the cumulative threshold is crossed,
        even if no single turn crossed the per-turn block threshold."""
        return await self._store.add_risk(
            session_id,
            risk_delta=risk_score,
            decay_per_second=self.limits.risk_decay_per_second,
            flag_threshold=self.limits.cumulative_risk_threshold,
            ttl_seconds=self.limits.session_ttl_seconds,
        )

    async def is_session_flagged(self, session_id: str) -> bool:
        return await self._store.is_flagged(session_id)

    async def reset_session(self, session_id: str) -> None:
        await self._store.reset_session(session_id)
