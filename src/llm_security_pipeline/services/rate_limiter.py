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

from ..resilience import (
    Degraded,
    RATE_LIMIT,
    SESSION_RISK,
    FailurePolicy,
    ResilientBackend,
)
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
    # Cumulative risk threshold for an ACTOR — an account, API key, tenant
    # or source address — accumulated across all of that actor's sessions.
    # Higher than the per-session threshold because it spans more traffic,
    # and longer-lived because rotating sessions is the evasion it exists
    # to catch. Session risk resets when the session id changes; this does
    # not, which is the whole point.
    actor_risk_threshold: float = 3.0
    actor_ttl_seconds: int = 86400


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

    def __init__(
        self,
        limits: SessionLimits | None = None,
        store: SessionStore | None = None,
        failure_policy: FailurePolicy | None = None,
        resilience: ResilientBackend | None = None,
    ):
        self.limits = limits or SessionLimits()
        # Defaults to an in-memory store, safe only for a single process.
        # Pass a RedisSessionStore explicitly for multi-process/instance deployments.
        self._store = store or InMemorySessionStore()
        # What happens when that store does not answer. Defaults to open
        # for both operations here: a counter being unavailable should not
        # take the product down. See resilience.py for the reasoning, and
        # note that every degraded call is reported, never silent.
        self._resilience = resilience or ResilientBackend(failure_policy)

    async def check_request(self, session_id: str, limits: SessionLimits | None = None) -> None:
        limits = limits or self.limits
        # Only the round trip goes inside run(): RateLimitExceeded below is
        # a decision about the traffic, not a backend failure, and must not
        # be mistaken for one.
        count = await self._resilience.run(
            RATE_LIMIT,
            lambda: self._store.increment_requests(session_id, limits.window_seconds),
        )
        if isinstance(count, Degraded):
            return
        if count > limits.max_requests_per_window:
            raise RateLimitExceeded(
                f"Session '{session_id}' exceeded {limits.max_requests_per_window} "
                f"requests within {limits.window_seconds}s."
            )

    async def check_tool_call(self, session_id: str, limits: SessionLimits | None = None) -> None:
        limits = limits or self.limits
        count = await self._resilience.run(
            RATE_LIMIT,
            lambda: self._store.increment_tool_calls(session_id, limits.window_seconds),
        )
        if isinstance(count, Degraded):
            return
        if count > limits.max_tool_calls_per_window:
            raise RateLimitExceeded(
                f"Session '{session_id}' exceeded {limits.max_tool_calls_per_window} "
                f"tool calls within {limits.window_seconds}s."
            )

    async def record_turn_risk(
        self, session_id: str, risk_score: float, actor_id: str | None = None,
    ) -> float:
        """Feed the per-turn risk score (e.g. from sanitizer.scan_text) into
        the session's cumulative total. Returns the updated cumulative
        score. Flags the session once the cumulative threshold is crossed,
        even if no single turn crossed the per-turn block threshold.

        With `actor_id`, the same score also accumulates against a second,
        coarser key. This is what survives session rotation: `session_id`
        is a string the caller supplies, so an attacker who sends a new one
        every turn keeps the session total at zero forever. An actor key
        does not have to be unforgeable to help — an account id, API key,
        tenant or source address only has to be more expensive to change
        than the session id is.
        """
        cumulative, _ = await self.record_turn_risk_and_check(session_id, risk_score, actor_id)
        return cumulative

    async def record_turn_risk_and_check(
        self, session_id: str, risk_score: float, actor_id: str | None = None,
        limits: SessionLimits | None = None,
    ) -> tuple[float, bool]:
        """record_turn_risk, plus whether the session or actor is flagged
        after the update.

        The store already knows the answer — it just decided it — so this
        reads it from the reply instead of asking again. On the session
        path that is one round trip fewer per turn with a session, two
        with an actor. `record_turn_risk` remains for callers that only
        want the number.
        """
        limits = limits or self.limits
        update = await self._resilience.run(
            SESSION_RISK,
            lambda: self._store.add_risk(
                session_id,
                risk_delta=risk_score,
                decay_per_second=limits.risk_decay_per_second,
                flag_threshold=limits.cumulative_risk_threshold,
                ttl_seconds=limits.session_ttl_seconds,
            ),
        )
        flagged = False if isinstance(update, Degraded) else update.flagged
        if actor_id is not None:
            actor_update = await self._resilience.run(
                SESSION_RISK,
                lambda: self._store.add_risk(
                    self.actor_key(actor_id),
                    risk_delta=risk_score,
                    decay_per_second=limits.risk_decay_per_second,
                    flag_threshold=limits.actor_risk_threshold,
                    ttl_seconds=limits.actor_ttl_seconds,
                ),
            )
            if not isinstance(actor_update, Degraded):
                flagged = flagged or actor_update.flagged
        # 0.0 rather than None: the caller feeds this into an audit record,
        # and a degraded accumulator has genuinely accumulated nothing. The
        # loss is reported through the degradation callback, not by
        # smuggling a sentinel into a float.
        cumulative = 0.0 if isinstance(update, Degraded) else update.cumulative
        return cumulative, flagged

    @staticmethod
    def actor_key(actor_id: str) -> str:
        """Namespaced so an actor's accumulator can never collide with a
        session id that happens to look like it."""
        return f"actor::{actor_id}"

    async def is_session_flagged(self, session_id: str, actor_id: str | None = None) -> bool:
        if await self._is_flagged(session_id):
            return True
        if actor_id is not None:
            return await self._is_flagged(self.actor_key(actor_id))
        return False

    async def is_actor_flagged(self, actor_id: str) -> bool:
        return await self._is_flagged(self.actor_key(actor_id))

    async def _is_flagged(self, key: str) -> bool:
        # Fail-open here reads as "not flagged", which is the permissive
        # answer by construction: the flag cannot be read, so nothing it
        # would have blocked is blocked.
        flagged = await self._resilience.run(
            SESSION_RISK, lambda: self._store.is_flagged(key), fallback=False,
        )
        return bool(flagged)

    async def reset_actor(self, actor_id: str) -> None:
        await self._store.reset_session(self.actor_key(actor_id))

    async def reset_session(self, session_id: str) -> None:
        await self._store.reset_session(session_id)
