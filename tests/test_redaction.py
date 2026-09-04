"""
Regression tests for redaction correctness.

Two bugs lived here, and both ended the same way: text the guard had
already detected as a secret was handed to the user anyway.

1. `find_secrets` used `findall`, which reports capture GROUPS rather than
   the whole match as soon as the pattern has any. A pattern with two
   groups produced tuples, and the old fallback resolved each of them with
   `pattern.search(text)` — which restarts from position 0 and therefore
   returned the first match for every match in the text. Every occurrence
   after the first was redacted with the wrong literal, i.e. not redacted.
   A pattern with exactly ONE group had a quieter version: `findall`
   returned the bare group, which is a `str` and so passed the type check,
   and only the group was replaced while the identifying prefix survived.

2. `redact` replaced each finding with `str.replace`, so a finding that is
   a substring of another corrupted the longer redaction depending on which
   category the dict happened to iterate first.

Custom patterns are the whole point of `pattern_config_path`, so neither
bug needed anything exotic to trigger — just a user writing a perfectly
ordinary regex with a group in it.
"""

from __future__ import annotations

import json

import pytest

from llm_security_pipeline import OutputGuard


def _guard(tmp_path, secret_patterns, pii_patterns=None):
    config = {
        "secret_patterns": secret_patterns,
        "pii_patterns": pii_patterns or [],
        "injection_patterns": [],
    }
    path = tmp_path / "patterns.json"
    path.write_text(json.dumps(config))
    return OutputGuard(pattern_config_path=str(path), include_default_patterns=False)


@pytest.mark.parametrize(
    "pattern",
    [
        r"sk-\w{6}",          # no capture group
        r"sk-(\w{6})",        # one group: findall returned the group alone
        r"(sk)-(\w{6})",      # two groups: findall returned tuples
    ],
    ids=["no_groups", "one_group", "two_groups"],
)
def test_every_occurrence_is_found_and_redacted_whatever_the_groups(tmp_path, pattern):
    guard = _guard(tmp_path, [{"name": "tok", "pattern": pattern, "flags": []}])
    text = "primo sk-AAAAAA e secondo sk-BBBBBB"

    result = guard.scan(text)

    assert result.secret_findings["tok"] == ["sk-AAAAAA", "sk-BBBBBB"]
    # The point of the test: no fragment of either secret survives.
    assert "AAAAAA" not in result.redacted_text
    assert "BBBBBB" not in result.redacted_text
    assert result.redacted_text == "primo [REDACTED:TOK] e secondo [REDACTED:TOK]"
    assert result.blocked is True


def test_group_pattern_does_not_leave_the_prefix_behind(tmp_path):
    """The one-group case used to redact only the group, leaving `key-`."""
    guard = _guard(tmp_path, [{"name": "api_key", "pattern": r"key-(\w{6})", "flags": []}])

    redacted = guard.scan("here is key-SECRET for you").redacted_text

    assert redacted == "here is [REDACTED:API_KEY] for you"


def test_repeated_identical_secret_is_redacted_everywhere(tmp_path):
    guard = _guard(tmp_path, [{"name": "tok", "pattern": r"(tk)-(\d{4})", "flags": []}])

    redacted = guard.scan("tk-1111 then tk-1111 then tk-1111").redacted_text

    assert "1111" not in redacted
    assert redacted.count("[REDACTED:TOK]") == 3


def test_overlapping_findings_do_not_corrupt_each_other(tmp_path):
    """A short finding inside a longer one must not break the longer one.

    With `str.replace` the outcome depended on dict iteration order: redact
    the inner match first and the outer literal no longer exists in the
    text, so the outer match silently survives.
    """
    guard = _guard(
        tmp_path,
        secret_patterns=[{"name": "long_token", "pattern": r"tok_[a-z]+_[0-9]{4}", "flags": []}],
        pii_patterns=[{"name": "fragment", "pattern": r"[a-z]+_[0-9]{4}", "flags": []}],
    )

    result = guard.scan("value tok_abcdef_1234 end")

    # The widest match wins, and secrets outrank PII on an exact tie.
    assert result.redacted_text == "value [REDACTED:LONG_TOKEN] end"
    assert "abcdef" not in result.redacted_text
    assert "1234" not in result.redacted_text


def test_redact_helper_is_position_based(tmp_path):
    """The public `redact(text, findings)` helper takes literals rather than
    spans, so it resolves them by position and applies the longest first."""
    text = "value tok_abcdef_1234 end"
    findings = {"fragment": ["abcdef_1234"], "long_token": ["tok_abcdef_1234"]}

    assert OutputGuard.redact(text, findings) == "value [REDACTED:LONG_TOKEN] end"


def test_phone_span_does_not_swallow_surrounding_words():
    guard = OutputGuard()

    redacted = guard.scan("call +39 340 1234567 now").redacted_text

    assert redacted == "call [REDACTED:PHONE_CANDIDATE] now"


def test_oversized_output_is_withheld_not_scanned():
    guard = OutputGuard(max_scan_chars=1_000)

    result = guard.scan("x" * 1_001)

    assert result.oversized is True
    assert result.blocked is True
    # Nothing was scanned, so nothing may be reported as clean, and the
    # unscanned text must not be what the caller forwards.
    assert result.secret_findings == {}
    assert "x" * 100 not in result.redacted_text


def test_max_scan_chars_none_disables_the_cap():
    guard = OutputGuard(max_scan_chars=None)

    assert guard.scan("x" * 300_000).oversized is False


def test_max_scan_chars_must_be_positive():
    with pytest.raises(ValueError):
        OutputGuard(max_scan_chars=0)
