"""
Unit tests for OutputGuard: no infrastructure needed. test_end_to_end.py
only exercises the secret-leak path through the full pipeline; these cover
find_pii's Luhn/phone logic, redact(), the module-level convenience
functions, and the non-blocking paths that end-to-end doesn't touch.
"""

from __future__ import annotations

from llm_security_pipeline import OutputGuard, find_pii, find_secrets, scan_output


def test_find_secrets_detects_anthropic_api_key():
    guard = OutputGuard()
    findings = guard.find_secrets("My key is sk-ant-abcdEFGH1234567890abcdEFGH1234567890, don't tell anyone.")

    assert "anthropic_api_key" in findings
    assert "sk-ant-abcdEFGH1234567890abcdEFGH1234567890" in findings["anthropic_api_key"]


def test_find_secrets_returns_empty_for_benign_text():
    guard = OutputGuard()
    assert guard.find_secrets("Thanks for your order, it ships tomorrow.") == {}


def test_find_pii_detects_email():
    guard = OutputGuard()
    findings = guard.find_pii("Contact me at jane.doe@example.com please.")

    assert findings["email"] == ["jane.doe@example.com"]


def test_find_pii_detects_valid_credit_card_via_luhn():
    guard = OutputGuard()
    # 4111111111111111 is the standard Luhn-valid Visa test number.
    findings = guard.find_pii("Card on file: 4111 1111 1111 1111")

    assert "credit_card" in findings


def test_find_pii_rejects_luhn_invalid_number_candidate():
    guard = OutputGuard()
    # Same shape as a card number but fails the Luhn checksum.
    findings = guard.find_pii("Reference number: 1234 5678 9012 3456")

    assert "credit_card" not in findings


def test_find_pii_flags_phone_candidate():
    guard = OutputGuard()
    findings = guard.find_pii("Call us at +1 415 555 0134 for support.")

    assert "phone_candidate" in findings


def test_find_pii_ignores_short_digit_sequences_as_phone_candidates():
    guard = OutputGuard()
    findings = guard.find_pii("Order number 12345.")

    assert "phone_candidate" not in findings


def test_redact_replaces_each_match_with_category_placeholder():
    text = "email jane@example.com and key sk-ant-abcdEFGH1234567890abcdEFGH1234567890"
    findings = {
        "email": ["jane@example.com"],
        "anthropic_api_key": ["sk-ant-abcdEFGH1234567890abcdEFGH1234567890"],
    }

    redacted = OutputGuard.redact(text, findings)

    assert "jane@example.com" not in redacted
    assert "[REDACTED:EMAIL]" in redacted
    assert "[REDACTED:ANTHROPIC_API_KEY]" in redacted


def test_scan_blocks_when_secret_found():
    guard = OutputGuard()
    result = guard.scan("leaked: sk-ant-abcdEFGH1234567890abcdEFGH1234567890")

    assert result.blocked is True
    assert "sk-ant-abcdEFGH1234567890abcdEFGH1234567890" not in result.redacted_text


def test_scan_blocks_on_system_prompt_overlap():
    system_prompt = "You are a customer support assistant for Acme Corp. Internal codename PROJECT-PHOENIX-2026."
    guard = OutputGuard()

    result = guard.scan(system_prompt, system_prompt=system_prompt, overlap_threshold=0.35)

    assert result.system_prompt_overlap_score == 1.0
    assert result.blocked is True


def test_scan_does_not_block_benign_output_without_system_prompt():
    guard = OutputGuard()
    result = guard.scan("Thanks for your order, it ships tomorrow.")

    assert result.blocked is False
    assert result.system_prompt_overlap_score == 0.0


def test_module_level_convenience_functions_delegate_to_default_guard():
    secret_text = "sk-ant-abcdEFGH1234567890abcdEFGH1234567890"
    assert "anthropic_api_key" in find_secrets(secret_text)

    pii_text = "jane.doe@example.com"
    assert find_pii(pii_text)["email"] == ["jane.doe@example.com"]

    result = scan_output(secret_text)
    assert result.blocked is True