"""
Unit tests for capability-token signing key rotation.

The property being tested is not "a keyring holds keys" but "there is no
moment during a rotation at which a valid token is rejected". With a single
key there is no such moment to find: the instant one process starts signing
with a new key, every token already in flight becomes a signature error
everywhere else, and the only way to change a key is a window of failures.

So the tests walk the three-step procedure and assert that tokens issued at
each step still verify at the next, and that the two ways of getting it
wrong — promoting before the new key is everywhere, retiring before the old
tokens have expired — fail loudly and say which key they mean.
"""

from __future__ import annotations

import secrets

import pytest

from llm_security_pipeline import (
    CapabilityToken,
    ScopeError,
    ScopeGuard,
    SigningKeyring,
    UnknownKeyId,
)

OLD = secrets.token_bytes(32)
NEW = secrets.token_bytes(32)


def _guard(keyring: SigningKeyring) -> ScopeGuard:
    return ScopeGuard(keyring=keyring)


# ---------------------------------------------------------------------------
# The rotation itself
# ---------------------------------------------------------------------------


async def test_rotation_never_rejects_a_valid_token():
    """The whole point, walked end to end.

    Each step is a deploy. A token issued before a step must still be
    spendable after it, because it is sitting in a queue or a retry while
    the deploy happens.
    """
    step0 = SigningKeyring.single(OLD, kid="2026-01")
    token_from_step0 = _guard(step0).issue_token("bot", ["read_crm"])

    # Step 1: everyone accepts the new key; nothing signs with it yet.
    step1 = step0.with_key("2026-02", NEW)
    assert step1.active == "2026-01"
    await _guard(step1).authorize(token_from_step0, "read_crm")
    token_from_step1 = _guard(step1).issue_token("bot", ["read_crm"])
    assert token_from_step1.key_id == "2026-01"

    # Step 2: promote. Old tokens are still in flight and must still work.
    step2 = step1.with_active("2026-02")
    await _guard(step2).authorize(token_from_step0, "read_crm")
    await _guard(step2).authorize(token_from_step1, "read_crm")
    token_from_step2 = _guard(step2).issue_token("bot", ["read_crm"])
    assert token_from_step2.key_id == "2026-02"

    # A process that has not restarted yet is still on step 1, and has to
    # accept what the promoted one is signing.
    await _guard(step1).authorize(token_from_step2, "read_crm")

    # Step 3: retire, once the longest TTL has elapsed.
    step3 = step2.without_key("2026-01")
    await _guard(step3).authorize(token_from_step2, "read_crm")


async def test_retiring_a_key_invalidates_tokens_that_named_it():
    """Step 3 done too early. It must fail, and name the key."""
    keyring = SigningKeyring.single(OLD, kid="2026-01").with_key("2026-02", NEW)
    old_token = _guard(keyring).issue_token("bot", ["read_crm"])
    retired = keyring.with_active("2026-02").without_key("2026-01")

    with pytest.raises(UnknownKeyId, match="2026-01"):
        await _guard(retired).authorize(old_token, "read_crm")


async def test_promoting_before_rollout_is_what_the_error_describes():
    """Step 2 before step 1 has finished: a process still on the old
    keyring receives a token signed with a key it has never heard of."""
    not_yet_updated = SigningKeyring.single(OLD, kid="2026-01")
    already_promoted = SigningKeyring(keys={"2026-01": OLD, "2026-02": NEW}, active="2026-02")

    token = _guard(already_promoted).issue_token("bot", ["read_crm"])

    with pytest.raises(UnknownKeyId, match="2026-02"):
        await _guard(not_yet_updated).authorize(token, "read_crm")


async def test_rotation_survives_serialization():
    """Tokens cross processes as strings, so the key id has to survive the
    round trip — it is inside the signed payload, not alongside it."""
    issuer = _guard(SigningKeyring(keys={"a": OLD, "b": NEW}, active="b"))
    wire = issuer.issue_token("bot", ["read_crm"]).to_str()

    parsed = CapabilityToken.from_str(wire)
    assert parsed.key_id == "b"
    await _guard(SigningKeyring(keys={"a": OLD, "b": NEW}, active="a")).authorize(parsed, "read_crm")


# ---------------------------------------------------------------------------
# The key id is untrusted input
# ---------------------------------------------------------------------------


async def test_repointing_the_key_id_does_not_help_an_attacker():
    """`kid` is inside the signed payload, so editing it breaks the
    signature — and even if it did not, the attacker would still have to
    produce a signature under the key they repointed at."""
    keyring = SigningKeyring(keys={"weak": OLD, "strong": NEW}, active="strong")
    token = _guard(keyring).issue_token("bot", ["read_crm"])
    payload_hex, _, signature = token.to_str().partition(".")

    edited = bytes.fromhex(payload_hex).replace(b'"strong"', b'"weak"  ')
    with pytest.raises(ScopeError):
        await _guard(keyring).authorize(
            CapabilityToken.from_str(f"{edited.hex()}.{signature}"), "read_crm"
        )


def test_token_without_a_key_id_is_rejected_at_parse():
    payload = (
        b'{"agent_id":"bot","scopes":["read_crm"],"expires_at":1e12,'
        b'"nonce":"aa","max_uses":1}'
    )
    with pytest.raises(ScopeError, match="kid"):
        CapabilityToken.from_str(f"{payload.hex()}.deadbeef")


async def test_unknown_key_id_is_a_lookup_not_a_search():
    """A miss must not be answered by trying every key in turn: that would
    make each verification a search over the keyring and make a retired key
    behave like a current one."""
    keyring = SigningKeyring(keys={"a": OLD, "b": NEW}, active="a")
    token = _guard(SigningKeyring.single(NEW, kid="c")).issue_token("bot", ["read_crm"])

    # NEW is in the keyring under the id "b", so a try-every-key
    # implementation would accept this. It must not.
    with pytest.raises(UnknownKeyId):
        await _guard(keyring).authorize(token, "read_crm")


# ---------------------------------------------------------------------------
# Keyring validation
# ---------------------------------------------------------------------------


def test_short_keys_are_refused():
    with pytest.raises(ValueError, match="at least 32"):
        SigningKeyring.single(b"too-short")


def test_active_key_must_be_in_the_keyring():
    with pytest.raises(ValueError, match="not in keys"):
        SigningKeyring(keys={"a": OLD}, active="b")


def test_empty_keyring_is_refused():
    with pytest.raises(ValueError, match="at least one key"):
        SigningKeyring(keys={}, active="a")


@pytest.mark.parametrize("kid", ["", "a" * 65, "has space", "new\nline", "unicodè"])
def test_key_ids_are_restricted_to_a_safe_alphabet(kid):
    """Key ids reach JSON payloads, error messages and audit log lines."""
    with pytest.raises(ValueError, match="key id"):
        SigningKeyring(keys={kid: OLD}, active=kid)


def test_cannot_promote_a_key_that_is_not_accepted_yet():
    with pytest.raises(ValueError, match="roll that out everywhere first"):
        SigningKeyring.single(OLD, kid="a").with_active("b")


def test_cannot_retire_the_active_key():
    with pytest.raises(ValueError, match="active signing key"):
        SigningKeyring.single(OLD, kid="a").with_key("b", NEW).without_key("a")


def test_keyring_is_immutable_once_a_guard_holds_it():
    """A caller keeping a reference to the mapping they passed in must not
    be able to add a key to a running guard."""
    mutable = {"a": OLD}
    keyring = SigningKeyring(keys=mutable, active="a")
    mutable["b"] = NEW

    assert keyring.key_ids == ("a",)
    with pytest.raises(TypeError):
        keyring.keys["c"] = NEW  # type: ignore[index]


def test_rotation_helpers_do_not_mutate_the_original():
    original = SigningKeyring.single(OLD, kid="a")

    original.with_key("b", NEW).with_active("b")

    assert original.key_ids == ("a",)
    assert original.active == "a"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_secret_key_is_shorthand_for_a_one_key_keyring():
    guard = ScopeGuard(secret_key=OLD)

    assert guard.active_key_id == "default"
    assert guard.issue_token("bot", ["read_crm"]).key_id == "default"


def test_secret_key_and_keyring_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        ScopeGuard(secret_key=OLD, keyring=SigningKeyring.single(NEW))


def test_pipeline_accepts_a_keyring():
    from llm_security_pipeline import SecurityPipeline

    keyring = SigningKeyring(keys={"a": OLD, "b": NEW}, active="b")
    pipeline = SecurityPipeline(session_identity="untrusted", scope_keyring=keyring)

    assert pipeline.scope_guard.active_key_id == "b"


def test_pipeline_refuses_two_ways_of_saying_the_same_thing():
    from llm_security_pipeline import SecurityPipeline

    with pytest.raises(ValueError, match="not both"):
        SecurityPipeline(
            session_identity="untrusted",
            scope_secret_key=OLD,
            scope_keyring=SigningKeyring.single(NEW),
        )


async def test_audit_log_records_which_key_a_token_was_issued_under():
    """Retiring a key safely is a question about the audit log: has any
    token naming it been spent recently? That only works if it is logged."""
    from llm_security_pipeline import SecurityPipeline
    from llm_security_pipeline.pipeline import AuditLogger

    events: list[tuple[str, dict]] = []

    class Collector(AuditLogger):
        async def log(self, event_type: str, data: dict) -> None:
            events.append((event_type, data))

    keyring = SigningKeyring.single(OLD, kid="2026-01")
    pipeline = SecurityPipeline(
        session_identity="untrusted", scope_keyring=keyring, audit_logger=Collector(),
    )
    token = pipeline.scope_guard.issue_token("bot", ["read_crm"])

    await pipeline.authorized_tool_call(token, "read_crm", lambda: "ok")

    tool_calls = [data for kind, data in events if kind == "tool_call"]
    assert tool_calls[-1]["token_key_id"] == "2026-01"
    assert tool_calls[-1]["status"] == "allowed"
