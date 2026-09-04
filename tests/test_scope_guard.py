"""
Unit tests for ScopeGuard: no infrastructure needed (InMemoryNonceStore is
exactly what a single-process test wants). The cross-process tests in
test_token_replay.py exercise the happy path plus replay under real
concurrency; these exercise every DENIAL branch of authorize(), plus
guarded_call(), none of which the cross-process tests touch.
"""

from __future__ import annotations

import time

import pytest

from llm_security_pipeline import ScopeError, ScopeGuard


def test_issue_token_sorts_and_dedupes_scopes():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm", "read_crm", "send_email"])

    assert token.payload["scopes"] == ["read_crm", "send_email"]
    assert token.payload["agent_id"] == "sales_bot"
    assert token.payload["max_uses"] == 1


async def test_authorize_allows_valid_token_within_scope():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"])

    await guard.authorize(token, "read_crm")  # must not raise


async def test_authorize_rejects_action_outside_granted_scope():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"])

    with pytest.raises(ScopeError, match="not authorized"):
        await guard.authorize(token, "send_email")


async def test_authorize_rejects_expired_token():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"], ttl_seconds=1)
    token.payload["expires_at"] = time.time() - 1  # force expiry without sleeping
    token.signature = guard._sign(token.payload)  # re-sign so this is a valid-but-expired token, not a tampered one

    with pytest.raises(ScopeError, match="expired"):
        await guard.authorize(token, "read_crm")


async def test_authorize_rejects_tampered_payload():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"])
    token.payload["scopes"].append("send_email")  # mutate after issuance, signature now stale

    with pytest.raises(ScopeError, match="tampering"):
        await guard.authorize(token, "send_email")


async def test_authorize_rejects_token_signed_with_different_key():
    guard_a = ScopeGuard()
    guard_b = ScopeGuard()
    token = guard_a.issue_token(agent_id="sales_bot", scopes=["read_crm"])

    with pytest.raises(ScopeError, match="tampering"):
        await guard_b.authorize(token, "read_crm")


async def test_authorize_rejects_replay_beyond_max_uses():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"], max_uses=1)

    await guard.authorize(token, "read_crm")  # first use: allowed
    with pytest.raises(ScopeError, match="replay"):
        await guard.authorize(token, "read_crm")  # second use: replay


async def test_guarded_call_executes_sync_func_when_authorized():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"])

    result = await guard.guarded_call(token, "read_crm", lambda x: x * 2, 21)

    assert result == 42


async def test_guarded_call_executes_async_func_when_authorized():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"])

    async def fetch(x):
        return x + 1

    result = await guard.guarded_call(token, "read_crm", fetch, 41)

    assert result == 42


async def test_guarded_call_raises_and_does_not_execute_when_unauthorized():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"])
    calls = []

    with pytest.raises(ScopeError):
        await guard.guarded_call(token, "send_email", lambda: calls.append(1))

    assert calls == []


def test_capability_token_to_str_roundtrip_shape():
    guard = ScopeGuard()
    token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"])

    encoded = token.to_str()
    payload_hex, _, signature = encoded.partition(".")

    assert signature == token.signature
    assert bytes.fromhex(payload_hex)  # decodes cleanly as hex