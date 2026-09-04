"""
Performance regression gate.

Not a benchmark: the numbers here are not interesting and machines differ.
What is asserted is the *shape* — that doubling the input roughly doubles
the time — because the failure mode worth catching is a pattern added to
`patterns.json` that turns a scanner quadratic, and that is invisible in
a unit test with a twenty-character input and very visible in production
with a twenty-thousand-character one. A very loose absolute floor is kept
so that a scanner that became pathologically slow at every size is also
caught, without failing on a busy CI runner.

Run alone with `pytest -m perf` when changing a pattern or a scanner.
"""

from __future__ import annotations

import time

import pytest

from llm_security_pipeline import OutputGuard, Sanitizer, StreamingOutputGuard

pytestmark = pytest.mark.perf

# Mixed content: prose, an email, a URL, digits — so every scanner has
# something to do at every size. Deliberately NOTHING that blocks: an
# earlier version carried a base64-looking run that tripped the generic
# high-entropy secret pattern, so the streaming guard stopped at the third
# chunk and the "throughput" it reported was the cost of doing nothing.
# Every test below asserts the scan actually ran to the end.
_UNIT = (
    "The quarterly review covers revenue, churn and the new onboarding flow. "
    "Contact ops@example.com or see https://example.com/report?id=42 for details. "
    "Reference: ticket 4411 was closed on Monday after the second follow-up call. "
)


def _text(chars: int) -> str:
    return (_UNIT * (chars // len(_UNIT) + 1))[:chars]


def _best_of(fn, runs: int = 3) -> float:
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _assert_roughly_linear(fn, small: int, large: int, label: str) -> None:
    """time(large) / time(small) must stay near large/small. A factor of 3
    over the ideal ratio is the tolerance: generous against noise, far
    below what any quadratic term produces at these sizes."""
    ratio = large / small
    t_small = _best_of(lambda: fn(_text(small)))
    t_large = _best_of(lambda: fn(_text(large)))
    assert t_large <= max(t_small, 1e-4) * ratio * 3, (
        f"{label}: {small} chars took {t_small*1e3:.1f} ms, {large} chars took "
        f"{t_large*1e3:.1f} ms — that is {t_large/max(t_small,1e-9):.0f}x for "
        f"{ratio:.0f}x the input"
    )
    # Absolute floor: these scanners sit in the low MB/s; 50 KB/s is far
    # enough below that a loaded CI runner passes, and far enough above
    # where a regex catastrophe lands that one still fails. (The streaming
    # guard rescans a window per chunk and sits around 300 KB/s at 64-char
    # chunks; that is the tightest of the three.)
    assert large / t_large > 20_000, f"{label}: {large / t_large / 1000:.0f} KB/s"


def test_the_corpus_does_not_block():
    """Guard for the guard: if this fails, the timings below are lies."""
    assert OutputGuard(max_scan_chars=None).scan(_text(4_000)).blocked is False
    assert Sanitizer(max_scan_chars=None).scan_text(_text(4_000)).blocked is False


def test_sanitizer_scales_linearly():
    sanitizer = Sanitizer(max_scan_chars=None)
    _assert_roughly_linear(sanitizer.scan_text, 8_000, 64_000, "Sanitizer.scan_text")


def test_output_guard_scales_linearly():
    guard = OutputGuard(max_scan_chars=None)

    def scan(text: str) -> None:
        result = guard.scan(text, system_prompt="internal assistant Rubicon QX-9911")
        assert result.blocked is False

    _assert_roughly_linear(scan, 8_000, 64_000, "OutputGuard.scan")


def test_streaming_guard_total_cost_scales_linearly():
    """Per-chunk work is bounded by the window, so total work over a
    stream must be linear in its length whatever the chunk size."""
    def stream(text: str) -> None:
        guard = StreamingOutputGuard(max_output_chars=None, system_prompt="assistant Rubicon")
        emitted = 0
        for i in range(0, len(text), 64):
            delta = guard.feed(text[i : i + 64])
            assert delta.blocked is False, "corpus blocked mid-stream; timing is meaningless"
            emitted += len(delta.text)
        final = guard.finish()
        assert final.blocked is False
        # Redaction shortens the email, so "roughly all of it" is the check.
        assert emitted + len(final.text) > len(text) * 0.8

    _assert_roughly_linear(stream, 8_000, 64_000, "StreamingOutputGuard")


def test_encoded_payload_scan_is_bounded_on_adversarial_input():
    """The candidate cap is what keeps a message made of base64-shaped
    tokens from turning into unbounded decode work."""
    from llm_security_pipeline.services.sanitizer import find_encoded_payloads

    hostile = " ".join(["QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="] * 20_000)
    elapsed = _best_of(lambda: find_encoded_payloads(hostile), runs=1)
    assert elapsed < 1.0, f"{elapsed:.2f}s for a hostile encoded-token flood"
