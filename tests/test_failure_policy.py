"""
Unit tests for what happens when the shared-state backend does not answer.

Two properties are being tested, and the second one is the one that makes
the first acceptable:

1. The policy is obeyed per operation. Fail-closed refuses; fail-open
   proceeds as though the check had not been configured.
2. Fail-open is never silent. Every degraded decision reaches the caller on
   the result object and reaches the log as a `backend_degraded` audit
   event, because a guard that stops guarding and says nothing is worse
   than no guard — "checked and clean" and "not checked" must not look
   alike.

The backends here are stores that raise or hang on purpose. No Redis is
involved: the question is what this library does with a failure, not
whether Redis produces one.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_security_pipeline import (
    BackendUnavailable,
    CapabilityToken,
    FailurePolicy,
    ResilientBackend,
    ScopeGuard,
    SecurityPipeline,
    SessionLimits,
    SessionRateLimiter,
    SigningKeyring,
)
from llm_security_pipeline.pipeline import AuditLogger
from llm_security_pipeline.resilience import DEGRADED, RATE_LIMIT, Degradation
from llm_security_pipeline.sessions.stores import (
    InMemoryNonceStore,
    InMemorySessionStore,
    NonceStore,
    SessionStore,
)

KEY = b"\x07" * 32


class Boom(Exception):
    """What a driver raises when the server is gone."""


class DeadSessionStore(SessionStore):
    async def increment_requests(self, session_id, window_seconds):
        raise Boom("connection refused")

    async def increment_tool_calls(self, session_id, window_seconds):
        raise Boom("connection refused")

    async def add_risk(self, session_id, risk_delta, decay_per_second, flag_threshold, ttl_seconds):
        raise Boom("connection refused")

    async def is_flagged(self, session_id):
        raise Boom("connection refused")

    async def reset_session(self, session_id):
        raise Boom("connection refused")


class DeadNonceStore(NonceStore):
    async def check_and_increment(self, nonce, max_uses, ttl_seconds):
        raise Boom("connection refused")


class HangingNonceStore(NonceStore):
    """Worse than dead: it never answers at all."""

    async def check_and_increment(self, nonce, max_uses, ttl_seconds):
        await asyncio.sleep(30)
        return 1


class CollectingAudit(AuditLogger):
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def log(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))


# ---------------------------------------------------------------------------
# The policy is obeyed
# ---------------------------------------------------------------------------


async def test_rate_limit_fails_open_by_default():
    """Refusing all traffic because a counter is unavailable turns a
    degraded dependency into a total outage."""
    limiter = SessionRateLimiter(store=DeadSessionStore())

    await limiter.check_request("s1")  # does not raise
    await limiter.check_tool_call("s1")


async def test_rate_limit_can_be_made_fail_closed():
    limiter = SessionRateLimiter(
        store=DeadSessionStore(), failure_policy=FailurePolicy(rate_limit="closed"),
    )

    with pytest.raises(BackendUnavailable) as exc:
        await limiter.check_request("s1")

    assert exc.value.operation == "rate_limit"


async def test_token_replay_fails_closed_by_default():
    """Without the nonce store there is no max_uses, so a single-use token
    becomes unlimited. Refusing the call is the bounded outcome."""
    guard = ScopeGuard(secret_key=KEY, nonce_store=DeadNonceStore())
    token = guard.issue_token("bot", ["read_crm"])

    with pytest.raises(BackendUnavailable) as exc:
        await guard.authorize(token, "read_crm")

    assert exc.value.operation == "token_replay"


async def test_token_replay_can_be_made_fail_open():
    guard = ScopeGuard(
        secret_key=KEY,
        nonce_store=DeadNonceStore(),
        failure_policy=FailurePolicy(token_replay="open"),
    )
    token = guard.issue_token("bot", ["read_crm"], max_uses=1)

    # Spendable more than once, which is exactly what fail-open here means.
    await guard.authorize(token, "read_crm")
    await guard.authorize(token, "read_crm")


async def test_a_backend_failure_is_not_an_authorization_failure():
    """The guard must still refuse an out-of-scope action rather than
    reporting a backend problem, and must not consult the store to know
    that: scope is decided before the replay check."""
    guard = ScopeGuard(secret_key=KEY, nonce_store=DeadNonceStore())
    token = guard.issue_token("bot", ["read_crm"])

    with pytest.raises(Exception) as exc:
        await guard.authorize(token, "send_email")

    assert not isinstance(exc.value, BackendUnavailable)


async def test_domain_exceptions_are_not_mistaken_for_backend_failures():
    """RateLimitExceeded is a decision about traffic. If it were raised
    inside the guarded call it would be swallowed by a fail-open policy."""
    from llm_security_pipeline import RateLimitExceeded

    limiter = SessionRateLimiter(
        limits=SessionLimits(max_requests_per_window=1), store=InMemorySessionStore(),
    )

    await limiter.check_request("s1")
    with pytest.raises(RateLimitExceeded):
        await limiter.check_request("s1")


# ---------------------------------------------------------------------------
# Fail-open is not silent
# ---------------------------------------------------------------------------


async def test_degradation_reaches_the_caller():
    audit = CollectingAudit()
    pipeline = SecurityPipeline(
        session_identity="untrusted",
        state_backend=None,
        rate_limiter=SessionRateLimiter(store=DeadSessionStore()),
        audit_logger=audit,
    )

    result = await pipeline.pre_process("hello", session_id="s1")

    assert result.blocked is False
    assert result.degraded, "a guard did not run and the caller was not told"
    assert {d.operation for d in result.degraded} <= {"rate_limit", "session_risk"}
    assert all(d.decision == "open" for d in result.degraded)


async def test_degradation_is_audited():
    audit = CollectingAudit()
    pipeline = SecurityPipeline(
        session_identity="untrusted",
        rate_limiter=SessionRateLimiter(store=DeadSessionStore()),
        audit_logger=audit,
    )

    await pipeline.pre_process("hello", session_id="s1")

    degraded_events = [data for kind, data in audit.events if kind == "backend_degraded"]
    assert degraded_events
    assert degraded_events[0]["decision"] == "open"
    assert "Boom" in degraded_events[0]["reason"]


async def test_the_input_scan_event_names_what_did_not_run():
    audit = CollectingAudit()
    pipeline = SecurityPipeline(
        session_identity="untrusted",
        rate_limiter=SessionRateLimiter(store=DeadSessionStore()),
        audit_logger=audit,
    )

    await pipeline.pre_process("hello", session_id="s1")

    scan = next(data for kind, data in audit.events if kind == "input_scan")
    assert "rate_limit" in scan["degraded"]


async def test_a_healthy_request_reports_nothing_degraded():
    pipeline = SecurityPipeline(session_identity="untrusted", audit_logger=CollectingAudit())

    result = await pipeline.pre_process("hello", session_id="s1")

    assert result.degraded == ()


async def test_concurrent_requests_do_not_inherit_each_others_failures():
    """One pipeline serves many requests at once, so the per-call bucket
    has to be per-call and not per-pipeline."""
    audit = CollectingAudit()
    healthy = SecurityPipeline(session_identity="untrusted", audit_logger=audit)
    broken = SecurityPipeline(
        session_identity="untrusted",
        rate_limiter=SessionRateLimiter(store=DeadSessionStore()),
        audit_logger=audit,
    )

    ok, bad = await asyncio.gather(
        healthy.pre_process("hello", session_id="a"),
        broken.pre_process("hello", session_id="b"),
    )

    assert ok.degraded == ()
    assert bad.degraded


# ---------------------------------------------------------------------------
# Timeouts and the circuit breaker
# ---------------------------------------------------------------------------


async def test_a_hung_backend_times_out_rather_than_waiting():
    """A backend that never answers produces no failure to have a policy
    about; without a timeout the request simply hangs."""
    guard = ScopeGuard(
        secret_key=KEY,
        nonce_store=HangingNonceStore(),
        failure_policy=FailurePolicy(token_replay="open", timeout_seconds=0.05),
    )
    token = guard.issue_token("bot", ["read_crm"])

    await asyncio.wait_for(guard.authorize(token, "read_crm"), timeout=2.0)


async def test_the_breaker_stops_calling_a_backend_that_is_down():
    calls = 0

    class Counting(SessionStore):
        async def increment_requests(self, session_id, window_seconds):
            nonlocal calls
            calls += 1
            raise Boom("down")

        async def increment_tool_calls(self, session_id, window_seconds): ...
        async def add_risk(self, session_id, risk_delta, decay_per_second, flag_threshold, ttl_seconds): ...
        async def is_flagged(self, session_id): ...
        async def reset_session(self, session_id): ...

    limiter = SessionRateLimiter(
        store=Counting(), failure_policy=FailurePolicy(failure_threshold=3, recovery_seconds=60),
    )

    for _ in range(10):
        await limiter.check_request("s1")

    # Three failures trip it; the remaining seven never reach the store.
    assert calls == 3


async def test_the_breaker_lets_one_call_through_after_recovery_seconds():
    calls = 0

    async def failing():
        nonlocal calls
        calls += 1
        raise Boom("down")

    backend = ResilientBackend(
        FailurePolicy(rate_limit="open", failure_threshold=1, recovery_seconds=0.05)
    )

    await backend.run(RATE_LIMIT, failing)
    await backend.run(RATE_LIMIT, failing)
    assert calls == 1

    await asyncio.sleep(0.06)
    await backend.run(RATE_LIMIT, failing)
    assert calls == 2


async def test_a_recovered_backend_stops_being_reported_as_degraded():
    state = {"up": False}
    events: list[Degradation] = []

    async def flaky():
        if not state["up"]:
            raise Boom("down")
        return 1

    backend = ResilientBackend(
        FailurePolicy(rate_limit="open"), on_degraded=lambda e: events.append(e),
    )

    assert await backend.run(RATE_LIMIT, flaky) is DEGRADED
    state["up"] = True
    assert await backend.run(RATE_LIMIT, flaky) == 1
    assert len(events) == 1


async def test_cancellation_is_not_converted_into_a_security_decision():
    """A caller going away is not a backend failure, and must not be
    answered by applying a policy."""

    async def slow():
        await asyncio.sleep(10)

    backend = ResilientBackend(FailurePolicy(rate_limit="open", timeout_seconds=None))
    task = asyncio.create_task(backend.run(RATE_LIMIT, slow))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["token_replay", "rate_limit", "session_risk", "provenance", "audit"])
def test_every_operation_has_its_own_setting(field):
    policy = FailurePolicy(**{field: "closed"})
    assert policy.decision_for(field) == "closed"


def test_a_typo_in_a_policy_is_a_construction_error():
    with pytest.raises(ValueError, match="'open' or 'closed'"):
        FailurePolicy(rate_limit="fail-open")


def test_unknown_operation_is_rejected():
    with pytest.raises(ValueError, match="Unknown operation"):
        FailurePolicy().decision_for("nonsense")


def test_defaults_are_asymmetric_on_purpose():
    """Closed where refusing is bounded, open where refusing would take the
    product down. If someone flattens these, this test should argue."""
    policy = FailurePolicy()

    assert policy.token_replay == "closed"
    assert policy.provenance == "closed"
    assert policy.rate_limit == "open"
    assert policy.session_risk == "open"
    assert policy.audit == "open"


def test_all_open_warns(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        FailurePolicy.all_open()

    assert any("without a replay check" in r.message for r in caplog.records)


def test_all_closed_is_available_without_a_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        policy = FailurePolicy.all_closed()

    assert policy.rate_limit == "closed"
    assert caplog.records == []


@pytest.mark.parametrize(
    "kwargs",
    [{"timeout_seconds": 0}, {"failure_threshold": 0}, {"recovery_seconds": -1}],
)
def test_invalid_tuning_is_refused(kwargs):
    with pytest.raises(ValueError):
        FailurePolicy(**kwargs)


# ---------------------------------------------------------------------------
# Audit logging is itself a backend
# ---------------------------------------------------------------------------


async def test_a_failing_audit_logger_does_not_refuse_the_request():
    class DeadAudit(AuditLogger):
        async def log(self, event_type, data):
            raise Boom("stream unavailable")

    pipeline = SecurityPipeline(session_identity="untrusted", audit_logger=DeadAudit())

    result = await pipeline.pre_process("hello", session_id="s1")

    assert result.blocked is False
    assert any(d.operation == "audit" for d in result.degraded)


async def test_a_failing_audit_logger_can_be_made_to_refuse():
    class DeadAudit(AuditLogger):
        async def log(self, event_type, data):
            raise Boom("stream unavailable")

    pipeline = SecurityPipeline(
        session_identity="untrusted",
        audit_logger=DeadAudit(),
        failure_policy=FailurePolicy(audit="closed"),
    )

    with pytest.raises(BackendUnavailable):
        await pipeline.pre_process("hello", session_id="s1")


async def test_audit_failure_does_not_recurse():
    """The degradation report is written through the audit logger. When the
    audit logger is what failed, that has to stop rather than loop."""
    attempts = 0

    class DeadAudit(AuditLogger):
        async def log(self, event_type, data):
            nonlocal attempts
            attempts += 1
            raise Boom("stream unavailable")

    pipeline = SecurityPipeline(session_identity="untrusted", audit_logger=DeadAudit())
    await pipeline.pre_process("hello", session_id="s1")

    # Bounded: one attempt per real event, none for the failures themselves.
    assert attempts < 10


# ---------------------------------------------------------------------------
# Sharing one view of the backend
# ---------------------------------------------------------------------------


def test_the_pipeline_shares_one_resilient_backend_with_its_guards():
    """Otherwise each guard forms its own opinion about whether the backend
    is up, and the breakers trip at different times."""
    pipeline = SecurityPipeline(
        session_identity="untrusted", failure_policy=FailurePolicy(rate_limit="closed"),
    )

    assert pipeline.rate_limiter._resilience is pipeline._resilience
    assert pipeline.scope_guard._resilience is pipeline._resilience
    assert pipeline.ingest_guard.resilience is pipeline._resilience


async def test_healthy_backends_are_untouched_by_any_of_this():
    """The policy must be invisible on the happy path."""
    limiter = SessionRateLimiter(
        limits=SessionLimits(max_requests_per_window=2), store=InMemorySessionStore(),
    )
    guard = ScopeGuard(secret_key=KEY, nonce_store=InMemoryNonceStore())
    token = guard.issue_token("bot", ["read_crm"], max_uses=1)

    await limiter.check_request("s1")
    await guard.authorize(token, "read_crm")

    from llm_security_pipeline import ScopeError

    with pytest.raises(ScopeError, match="maximum allowed"):
        await guard.authorize(token, "read_crm")


async def test_rotation_and_failure_policy_compose():
    """Two features that both touch ScopeGuard construction."""
    keyring = SigningKeyring(keys={"a": KEY, "b": b"\x08" * 32}, active="b")
    guard = ScopeGuard(
        keyring=keyring,
        nonce_store=DeadNonceStore(),
        failure_policy=FailurePolicy(token_replay="open"),
    )

    token = CapabilityToken.from_str(guard.issue_token("bot", ["read_crm"]).to_str())
    await guard.authorize(token, "read_crm")

    assert token.key_id == "b"
