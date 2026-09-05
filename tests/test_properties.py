"""
Property-based tests over the parsers that read untrusted input.

Every hand-written test in this suite feeds the code an input the author
thought of. These feed it inputs nobody thought of, which for a security
library is the point: `from_str` parses whatever a client sent, the
sanitizer normalizes whatever a user typed, and the streaming guard sees
the model's output in whatever chunking the transport happened to produce.
"Never raises anything but the documented error" and "gives the same
answer regardless of chunking" are properties, and properties are what
Hypothesis is for.

These run with a modest example budget so CI stays fast; raise
`max_examples` locally when changing a parser.
"""

from __future__ import annotations

import secrets

from hypothesis import given, settings
from hypothesis import strategies as st

from llm_security_pipeline import (
    CapabilityToken,
    OutputGuard,
    Sanitizer,
    ScopeError,
    ScopeGuard,
    StreamingOutputGuard,
)
from llm_security_pipeline.services.sanitizer import find_encoded_payloads, find_hidden_text

BUDGET = settings(max_examples=150, deadline=None)

any_text = st.text(min_size=0, max_size=400)
# Includes the invisible/tag codepoints the sanitizer decodes on purpose.
sneaky_text = st.text(
    alphabet=st.one_of(
        st.characters(),
        st.sampled_from("\u200b\u200c\u200d\u2060\ufeff\u202e\u202d"),
        st.characters(min_codepoint=0xE0000, max_codepoint=0xE007F),
    ),
    max_size=300,
)


# ---------------------------------------------------------------------------
# Parsers never raise anything but their documented error
# ---------------------------------------------------------------------------


@BUDGET
@given(st.text(max_size=2000))
def test_from_str_only_ever_raises_scope_error(garbage):
    try:
        CapabilityToken.from_str(garbage)
    except ScopeError:
        pass


@BUDGET
@given(st.binary(max_size=600))
def test_from_str_on_arbitrary_bytes_only_raises_scope_error(raw):
    token = raw.hex() + "." + "ab" * 32
    try:
        CapabilityToken.from_str(token)
    except ScopeError:
        pass


@BUDGET
@given(st.text(max_size=300))
def test_a_real_token_survives_any_suffix_or_is_cleanly_rejected(suffix):
    guard = ScopeGuard(secret_key=secrets.token_bytes(32))
    wire = guard.issue_token("bot", ["read"]).to_str()
    try:
        CapabilityToken.from_str(wire + suffix)
    except ScopeError:
        pass


@BUDGET
@given(sneaky_text)
def test_the_sanitizer_never_raises(text):
    result = Sanitizer(max_scan_chars=None).scan_text(text)
    assert 0.0 <= result.risk_score <= 1.0


@BUDGET
@given(sneaky_text)
def test_hidden_text_and_encoded_payload_finders_never_raise(text):
    find_hidden_text(text)
    find_encoded_payloads(text)


@BUDGET
@given(sneaky_text)
def test_confusable_folding_is_total_and_idempotent(text):
    """Any text; folding twice folds nothing more; single-script words are
    left byte-identical."""
    from llm_security_pipeline.services.sanitizer import fold_confusables

    once, hits = fold_confusables(text)
    twice, more = fold_confusables(once)
    assert hits >= 0 and more == 0 and twice == once
    assert len(once) == len(text)   # one letter maps to one letter


@BUDGET
@given(any_text)
def test_the_output_guard_never_raises_and_never_leaves_a_finding(text):
    guard = OutputGuard(max_scan_chars=None)
    result = guard.scan(text)
    for findings in (result.secret_findings, result.pii_findings):
        for literals in findings.values():
            for literal in literals:
                assert literal not in result.redacted_text or literal == ""


# ---------------------------------------------------------------------------
# Streaming gives the same answer as buffering, whatever the chunking
# ---------------------------------------------------------------------------


@st.composite
def text_and_chunking(draw):
    text = draw(st.text(max_size=300))
    if not text:
        return text, []
    cuts = draw(st.lists(st.integers(1, max(1, len(text) - 1)), max_size=20))
    cuts = sorted({c for c in cuts if 0 < c < len(text)})
    bounds = [0, *cuts, len(text)]
    return text, [text[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


@BUDGET
@given(text_and_chunking())
def test_streaming_matches_buffered_for_any_chunking(case):
    """If these ever disagree, the transport has become a security
    decision."""
    text, chunks = case
    buffered = OutputGuard(max_scan_chars=None).scan(text)

    guard = StreamingOutputGuard(max_output_chars=None)
    emitted = []
    blocked = False
    for chunk in chunks:
        delta = guard.feed(chunk)
        if delta.blocked:
            blocked = True
            break
        emitted.append(delta.text)
    if not blocked:
        final = guard.finish()
        blocked = final.blocked
        if not blocked:
            emitted.append(final.text)

    assert blocked == buffered.blocked
    if not blocked:
        assert "".join(emitted) == buffered.redacted_text


@BUDGET
@given(text_and_chunking())
def test_streaming_never_emits_more_than_it_was_given(case):
    text, chunks = case
    guard = StreamingOutputGuard(max_output_chars=None, redact_pii=False)
    out = []
    for chunk in chunks:
        delta = guard.feed(chunk)
        if delta.blocked:
            return
        out.append(delta.text)
    final = guard.finish()
    if not final.blocked:
        out.append(final.text)
        assert "".join(out) == text
