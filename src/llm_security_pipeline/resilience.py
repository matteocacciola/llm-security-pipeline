"""
resilience.py
What the guards do when the shared-state backend does not answer.

Every cross-process guarantee in this library is a round trip to Redis,
PostgreSQL or MySQL. Those round trips fail: a failover, a connection pool
exhausted, a network partition, a maintenance window. Until now the
resulting driver exception simply propagated, which quietly made the
decision on the operator's behalf and made it the same way for every guard:
the request 500s.

That is a decision, not an absence of one, and it is the wrong default for
some of these operations and the right one for others. So it is named.

--- Why one global setting would be wrong ---------------------------------

The question "let it through or refuse it" has a different answer per
operation, because the two failure modes are not symmetrical in the same
direction each time:

* **Token replay** — if the nonce store is unreachable, `max_uses` cannot
  be honoured, and a token that should be spendable once becomes spendable
  without limit. Refusing a tool call is bounded and recoverable; an
  unbounded token is not. Defaults to **closed**.
* **Provenance** — an unverifiable retrieval is exactly the case the
  provenance store exists to distinguish from a verified one. Defaults to
  **closed**.
* **Rate limiting** — the budget mitigates cost and DoS abuse. Refusing all
  traffic because a counter is unavailable converts a degraded dependency
  into a total outage, which is a larger incident than the one being
  prevented. Defaults to **open**.
* **Session risk** — a heuristic accumulator supporting multi-turn
  detection. Losing it for the duration of an outage loses a supporting
  signal, not the defence. Defaults to **open**.
* **Audit logging** — a log write that fails should not refuse the request
  it was describing. Defaults to **open**, and the loss is itself logged.
* **Semantic detectors** — a supporting signal running out of process, on
  a model server that can be slow or down. Refusing traffic because a
  classifier is unreachable trades a signal for an outage. Defaults to
  **open**, and a detector that failed contributes nothing rather than
  contributing 0.0, which are not the same claim.

--- Why fail-open is only acceptable when it is visible -------------------

A guard that stops guarding and says nothing is worse than no guard, for
the same reason the media scanner refuses to report 0.0 without attaching
its extractor errors: "checked and clean" and "not checked" must not look
alike. So every degraded decision is logged, emitted as a `backend_degraded`
audit event, and marked on the result the caller receives
(`PreProcessResult.degraded`). Fail-open here means the request proceeds,
never that it proceeds silently.

--- Timeouts and the circuit breaker --------------------------------------

A hung backend is worse than a dead one: without a timeout the failure
never arrives and the request simply waits, so `timeout_seconds` bounds
every call. And once a backend is down, paying that timeout on every
request turns an availability problem into a latency problem for as long as
it lasts — hence the circuit breaker, which after `failure_threshold`
consecutive failures applies the policy immediately and retries the backend
once every `recovery_seconds`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

FAIL_OPEN = "open"
FAIL_CLOSED = "closed"
_DECISIONS = (FAIL_OPEN, FAIL_CLOSED)

# Operation categories. Named rather than free-form so a typo in a policy
# is a construction error instead of a setting that silently never applies.
TOKEN_REPLAY = "token_replay"
RATE_LIMIT = "rate_limit"
SESSION_RISK = "session_risk"
PROVENANCE = "provenance"
AUDIT = "audit"
DETECTOR = "detector"

OPERATIONS = (TOKEN_REPLAY, RATE_LIMIT, SESSION_RISK, PROVENANCE, AUDIT, DETECTOR)


class BackendUnavailable(Exception):
    """A guard could not reach its backend and the policy says to refuse.

    Carries the operation so a caller can map it to a response: this is a
    503, not a 403. The request was not denied on its merits, it was not
    evaluated.
    """

    def __init__(self, operation: str, cause: BaseException | None = None):
        self.operation = operation
        self.cause = cause
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        super().__init__(
            f"The {operation} check could not run because its backend was "
            f"unavailable{detail}. This deployment's failure policy for "
            f"{operation} is 'closed', so the request was refused rather than "
            "allowed through unchecked."
        )


@dataclass(frozen=True)
class Degradation:
    """One guard that did not run, and what was done about it."""

    operation: str
    decision: str
    reason: str
    circuit_open: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "decision": self.decision,
            "reason": self.reason,
            "circuit_open": self.circuit_open,
        }


@dataclass(frozen=True)
class FailurePolicy:
    """What each guard does when its backend is unreachable.

    The defaults are not a neutral position — there isn't one — but they
    are chosen on the same principle throughout: fail closed where refusing
    is bounded and letting through is not, fail open where refusing would
    turn a dependency outage into a product outage.

        # A deployment that would rather be down than unmetered
        FailurePolicy(rate_limit="closed", session_risk="closed")

        # A deployment that has decided availability wins everywhere
        FailurePolicy.all_open()   # logs a warning; read its docstring first
    """

    token_replay: str = FAIL_CLOSED
    provenance: str = FAIL_CLOSED
    rate_limit: str = FAIL_OPEN
    session_risk: str = FAIL_OPEN
    audit: str = FAIL_OPEN
    detector: str = FAIL_OPEN

    # A hung backend never produces a failure to have a policy about, so
    # every call is bounded. Generous enough not to trip on a slow query,
    # short enough that a partition does not become the user's problem.
    timeout_seconds: float | None = 2.0
    # Consecutive failures before the policy is applied without calling the
    # backend at all.
    failure_threshold: int = 5
    # How long to wait before letting one call through to test recovery.
    recovery_seconds: float = 10.0

    def __post_init__(self) -> None:
        for operation in OPERATIONS:
            decision = getattr(self, operation)
            if decision not in _DECISIONS:
                raise ValueError(
                    f"FailurePolicy.{operation} must be 'open' or 'closed', "
                    f"got {decision!r}."
                )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive, or None to wait forever.")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1.")
        if self.recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be positive.")

    def decision_for(self, operation: str) -> str:
        try:
            return getattr(self, operation)  # type: ignore[no-any-return]
        except AttributeError:
            raise ValueError(
                f"Unknown operation {operation!r}; expected one of {', '.join(OPERATIONS)}."
            ) from None

    @classmethod
    def all_closed(cls, **kwargs: Any) -> "FailurePolicy":
        """Refuse everything the backend cannot answer.

        Defensible, and worth understanding before choosing: a Redis blip
        now returns 503 for every request, including the ones that would
        have been perfectly safe to serve.
        """
        return cls(**dict.fromkeys(OPERATIONS, FAIL_CLOSED), **kwargs)  # type: ignore[arg-type]

    @classmethod
    def all_open(cls, **kwargs: Any) -> "FailurePolicy":
        """Serve everything the backend cannot answer.

        This makes a backend outage into a window with no replay
        protection and no provenance verification, which is the window an
        attacker would pick if they could pick one. Availability is
        sometimes worth that; choosing it accidentally never is.
        """
        logger.warning(
            "FailurePolicy.all_open(): while the state backend is unreachable, "
            "capability tokens will be accepted without a replay check and "
            "retrieved documents without provenance verification. Every such "
            "decision is logged as a backend_degraded audit event."
        )
        return cls(**dict.fromkeys(OPERATIONS, FAIL_OPEN), **kwargs)  # type: ignore[arg-type]


DEFAULT_FAILURE_POLICY = FailurePolicy()


class _CircuitBreaker:
    """One breaker per operation category.

    Deliberately not shared across categories: they fail together in
    practice, but a breaker that trips for one guard because another one
    was unlucky is hard to reason about during an incident.
    """

    def __init__(self, failure_threshold: int, recovery_seconds: float):
        self._threshold = failure_threshold
        self._recovery = recovery_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_tripped(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._recovery:
            # Half-open: let exactly one call through to test the backend.
            self._opened_at = None
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold:
            self._opened_at = time.monotonic()


class Degraded:
    """Sentinel type for "the call did not happen".

    Distinct from any value a store could legitimately return, including
    None and 0. A class rather than a bare object so callers can write
    `isinstance(result, Degraded)`, which narrows the union for a type
    checker where an identity check against a sentinel instance does not.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<DEGRADED>"


DEGRADED = Degraded()


class ResilientBackend:
    """Runs a backend call under a policy, a timeout and a breaker.

    One instance is normally shared by every guard in a pipeline, so the
    breakers and the degradation callback see one coherent picture. Guards
    construct their own from a policy when used standalone.
    """

    def __init__(
        self,
        policy: FailurePolicy | None = None,
        on_degraded: Callable[[Degradation], Awaitable[None] | None] | None = None,
    ):
        self.policy = policy or DEFAULT_FAILURE_POLICY
        self._on_degraded = on_degraded
        # Keyed by (operation, instance). Most operations have one
        # instance — there is one nonce store — but detectors have one
        # breaker EACH: a breaker shared across the category would let one
        # broken model server switch off every other detector alongside
        # it, which is a healthy signal lost to an unrelated failure.
        self._breakers: dict[tuple[str, str], _CircuitBreaker] = {}

    def with_callback(
        self, on_degraded: Callable[[Degradation], Awaitable[None] | None],
    ) -> "ResilientBackend":
        """Attach a reporting callback, keeping the same breakers."""
        self._on_degraded = on_degraded
        return self

    def _breaker(self, operation: str, instance: str) -> _CircuitBreaker:
        key = (operation, instance)
        breaker = self._breakers.get(key)
        if breaker is None:
            breaker = self._breakers[key] = _CircuitBreaker(
                self.policy.failure_threshold, self.policy.recovery_seconds,
            )
        return breaker

    async def run(
        self,
        operation: str,
        call: Callable[[], Awaitable[T]],
        fallback: T | Degraded = DEGRADED,
        instance: str = "",
    ) -> T | Degraded:
        """Run `call`, returning `fallback` if the backend fails and the
        policy for `operation` is open.

        `call` is a factory rather than an awaitable so that nothing is
        started when the breaker is already tripped. It must contain ONLY
        the backend round trip: a domain exception raised by the caller's
        own logic (over budget, out of scope) must not be mistaken for an
        unreachable backend.
        """
        decision = self.policy.decision_for(operation)
        breaker = self._breaker(operation, instance)

        if breaker.is_tripped:
            return await self._degrade(
                operation, decision, fallback,
                reason="circuit breaker open after repeated failures",
                cause=None, circuit_open=True,
            )

        try:
            if self.policy.timeout_seconds is None:
                return await call()
            return await asyncio.wait_for(call(), self.policy.timeout_seconds)
        except asyncio.CancelledError:
            # Not a backend failure: the caller went away. Cancellation must
            # never be converted into a security decision.
            raise
        except Exception as exc:
            breaker.record_failure()
            return await self._degrade(
                operation, decision, fallback,
                reason=f"{type(exc).__name__}: {exc}"[:300], cause=exc,
            )

    async def _degrade(
        self,
        operation: str,
        decision: str,
        fallback: T | Degraded,
        reason: str,
        cause: BaseException | None,
        circuit_open: bool = False,
    ) -> T | Degraded:
        event = Degradation(
            operation=operation, decision=decision, reason=reason, circuit_open=circuit_open,
        )
        logger.warning(
            "Backend unavailable for %s check (%s); failure policy is '%s'.",
            operation, reason, decision,
        )
        if self._on_degraded is not None:
            outcome = self._on_degraded(event)
            if outcome is not None and hasattr(outcome, "__await__"):
                await outcome

        if decision == FAIL_CLOSED:
            raise BackendUnavailable(operation, cause)
        return fallback
