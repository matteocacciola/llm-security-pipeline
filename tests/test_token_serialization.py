"""
Unit tests for capability-token serialization and key handling.

`to_str` existed without a counterpart, so a token could be written down
but never read back — in a deployment where tokens are issued by one
service and spent by another, that is the only thing you actually need.
`from_str` closes it, and the signature is verified over the bytes the
token arrived as rather than over a re-serialization of the parsed
payload, so verification does not depend on JSON round-tripping to
byte-identical output.

The other half is the key. A ScopeGuard with no `secret_key` generates one
per process, which is correct for a single process and quietly wrong for
any deployment with more than one: the same token verifies on the pod that
issued it and fails everywhere else, reported as tampering.
"""

from __future__ import annotations

import logging

import pytest

from llm_security_pipeline import CapabilityToken, ScopeError, ScopeGuard
from llm_security_pipeline.services.scope_guard import MAX_TOKEN_CHARS

KEY = b"\x01" * 32


def test_token_survives_a_round_trip_between_two_guards():
    """The deployment this library targets: issued in one process, spent in
    another, with only the serialized string in between."""
    issuer = ScopeGuard(secret_key=KEY)
    token = issuer.issue_token("bot", ["read_crm"], subject="user-1")

    parsed = CapabilityToken.from_str(token.to_str())

    assert parsed.payload == token.payload
    assert parsed.subject == "user-1"


async def test_parsed_token_authorizes_against_a_separate_guard():
    issuer = ScopeGuard(secret_key=KEY)
    token = issuer.issue_token("bot", ["read_crm"], subject="user-1")

    spender = ScopeGuard(secret_key=KEY)
    await spender.authorize(CapabilityToken.from_str(token.to_str()), "read_crm", subject="user-1")


async def test_parsed_token_keeps_its_scope_boundary():
    issuer = ScopeGuard(secret_key=KEY)
    token = issuer.issue_token("bot", ["read_crm"])
    parsed = CapabilityToken.from_str(token.to_str())

    with pytest.raises(ScopeError):
        await ScopeGuard(secret_key=KEY).authorize(parsed, "send_email")


async def test_parsed_token_keeps_its_signed_constraints():
    issuer = ScopeGuard(secret_key=KEY)
    token = issuer.issue_token("bot", ["read_crm"], constraints={"account_id": "42"})

    parsed = CapabilityToken.from_str(token.to_str())

    ScopeGuard.check_constraints(parsed, account_id="42")
    with pytest.raises(ScopeError):
        ScopeGuard.check_constraints(parsed, account_id="43")


async def test_edited_payload_fails_verification():
    """The serialized form is encoded, not encrypted — anyone can read and
    rewrite it. What must hold is that a rewrite stops verifying."""
    issuer = ScopeGuard(secret_key=KEY)
    token = issuer.issue_token("bot", ["read_crm"])
    payload_hex, _, signature = token.to_str().partition(".")

    edited = bytes.fromhex(payload_hex).replace(b"read_crm", b"send_mail")
    forged = f"{edited.hex()}.{signature}"

    with pytest.raises(ScopeError, match="signature"):
        await ScopeGuard(secret_key=KEY).authorize(
            CapabilityToken.from_str(forged), "send_mail"
        )


@pytest.mark.parametrize(
    "bad",
    ["", "nodot", ".", "abc.", ".abc", "zz.1234", "7b7d", "not-hex-at-all.deadbeef"],
)
def test_malformed_tokens_raise_scope_error_not_a_parse_error(bad):
    """One exception type for a caller handling untrusted input."""
    with pytest.raises(ScopeError):
        CapabilityToken.from_str(bad)


def test_non_object_payload_is_rejected():
    payload = b"[1, 2, 3]"
    with pytest.raises(ScopeError, match="not an object"):
        CapabilityToken.from_str(f"{payload.hex()}.deadbeef")


def test_payload_missing_required_fields_is_rejected():
    payload = b'{"agent_id": "bot"}'
    with pytest.raises(ScopeError, match="missing"):
        CapabilityToken.from_str(f"{payload.hex()}.deadbeef")


def test_oversized_token_is_refused_before_parsing():
    with pytest.raises(ScopeError, match="exceeds"):
        CapabilityToken.from_str("a" * (MAX_TOKEN_CHARS + 1))


def test_non_string_token_is_refused():
    with pytest.raises(ScopeError):
        CapabilityToken.from_str(b"bytes-not-str")  # type: ignore[arg-type]


def test_signature_is_verified_over_the_bytes_received():
    """Verification must not depend on re-serializing the parsed payload to
    byte-identical JSON — a property of one encoder, not of the format."""
    issuer = ScopeGuard(secret_key=KEY)
    token = issuer.issue_token("bot", ["read_crm"])
    parsed = CapabilityToken.from_str(token.to_str())

    assert parsed.raw is not None
    assert parsed.canonical() == parsed.raw
    # A freshly issued token has no wire form yet and falls back to the dict.
    assert token.raw is None
    assert token.canonical() == parsed.raw


def test_generated_key_warns_that_it_is_process_local(caplog):
    with caplog.at_level(logging.WARNING):
        ScopeGuard()

    assert any("secret_key" in record.message for record in caplog.records)


def test_explicit_key_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        ScopeGuard(secret_key=KEY)

    assert caplog.records == []


async def test_cross_process_key_mismatch_explains_itself():
    """Two guards that never agreed on a key are the multi-pod bug. The
    error must point at the key rather than accusing the token."""
    token = ScopeGuard(secret_key=KEY).issue_token("bot", ["read_crm"])

    with pytest.raises(ScopeError, match="per-process key"):
        await ScopeGuard().authorize(CapabilityToken.from_str(token.to_str()), "read_crm")
