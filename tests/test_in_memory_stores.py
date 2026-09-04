"""
Unit tests for the in-memory NonceStore/SessionStore implementations.

Unlike the cross-process integration tests, these need no external
infrastructure and no ProcessPoolExecutor: InMemoryNonceStore and
InMemorySessionStore are only ever correct within a single process, so a
single asyncio event loop is exactly the scenario they're meant for.
"""

from __future__ import annotations

import time

import pytest

from llm_security_pipeline import InMemoryNonceStore, InMemorySessionStore


async def test_nonce_store_increments_per_nonce():
    store = InMemoryNonceStore()

    assert await store.check_and_increment("nonce-a", max_uses=1, ttl_seconds=60) == 1
    assert await store.check_and_increment("nonce-a", max_uses=1, ttl_seconds=60) == 2
    # A different nonce has its own independent counter.
    assert await store.check_and_increment("nonce-b", max_uses=1, ttl_seconds=60) == 1


async def test_session_store_request_budget_resets_after_window():
    store = InMemorySessionStore()
    session_id = "session-1"

    assert await store.increment_requests(session_id, window_seconds=60) == 1
    assert await store.increment_requests(session_id, window_seconds=60) == 2

    # Simulate the window having elapsed by back-dating the recorded
    # events past the trailing edge of the window.
    store._requests[session_id] = type(store._requests[session_id])(
        ts - 61 for ts in store._requests[session_id]
    )

    assert await store.increment_requests(session_id, window_seconds=60) == 1


async def test_session_store_window_slides_rather_than_resets():
    """The property a fixed window lacks: "N per W" over EVERY interval
    of length W. Two events at t=0, one at t=59 -> 3 in window; at t=61
    the first two have aged out but the third has not."""
    store = InMemorySessionStore()
    session_id = "session-1"

    await store.increment_requests(session_id, window_seconds=60)
    await store.increment_requests(session_id, window_seconds=60)
    events = store._requests[session_id]
    now = time.time()
    events.clear()
    events.extend([now - 61, now - 61, now - 2])

    # Only the recent one survives, plus this call.
    assert await store.increment_requests(session_id, window_seconds=60) == 2


async def test_session_store_tool_calls_tracked_independently_of_requests():
    store = InMemorySessionStore()
    session_id = "session-1"

    await store.increment_requests(session_id, window_seconds=60)
    await store.increment_requests(session_id, window_seconds=60)
    assert await store.increment_tool_calls(session_id, window_seconds=60) == 1


async def test_session_store_risk_decays_over_time_and_flags_at_threshold():
    store = InMemorySessionStore()
    session_id = "session-1"

    updated = await store.add_risk(
        session_id, risk_delta=0.5, decay_per_second=0.0, flag_threshold=1.0, ttl_seconds=60,
    )
    assert updated.cumulative == pytest.approx(0.5)
    assert updated.flagged is False
    assert await store.is_flagged(session_id) is False

    updated = await store.add_risk(
        session_id, risk_delta=0.6, decay_per_second=0.0, flag_threshold=1.0, ttl_seconds=60,
    )
    assert updated.cumulative == pytest.approx(1.1)
    assert updated.flagged is True
    assert await store.is_flagged(session_id) is True


async def test_session_store_risk_decay_reduces_cumulative_value():
    store = InMemorySessionStore()
    session_id = "session-1"

    await store.add_risk(session_id, risk_delta=1.0, decay_per_second=1.0, flag_threshold=100.0, ttl_seconds=60)
    # Back-date the last-update timestamp to simulate 10 elapsed seconds of decay.
    current, _last_update = store._risk[session_id]
    store._risk[session_id] = (current, time.time() - 10)

    updated = await store.add_risk(
        session_id, risk_delta=0.0, decay_per_second=1.0, flag_threshold=100.0, ttl_seconds=60,
    )
    assert updated.cumulative == pytest.approx(0.0)
    assert updated.flagged is False


async def test_session_store_reset_clears_all_state():
    store = InMemorySessionStore()
    session_id = "session-1"

    await store.increment_requests(session_id, window_seconds=60)
    await store.increment_tool_calls(session_id, window_seconds=60)
    await store.add_risk(session_id, risk_delta=5.0, decay_per_second=0.0, flag_threshold=1.0, ttl_seconds=60)
    assert await store.is_flagged(session_id) is True

    await store.reset_session(session_id)

    assert await store.increment_requests(session_id, window_seconds=60) == 1
    assert await store.increment_tool_calls(session_id, window_seconds=60) == 1
    assert await store.is_flagged(session_id) is False