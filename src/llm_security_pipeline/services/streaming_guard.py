"""
streaming_guard.py
Scanning a model response while it is still being written.

`OutputGuard.scan` needs the finished text. Almost every deployed chat
surface streams instead, and a guard that only works on a complete response
is a guard that gets skipped — so this scans the response as it arrives.

Three problems have to be solved, and only the first is obvious.

--- 1. A secret can straddle a chunk boundary ------------------------------

`sk-ant-abc` arrives in one chunk and `def123` in the next. Scanning each
chunk on its own finds nothing in either. So the scan runs over a window,
not over the chunk: the unemitted buffer plus the tail of what was already
emitted, and matches are found across the seam.

--- 2. Emitted text cannot be recalled ------------------------------------

This is the real constraint, and it is what shapes the whole design. Once a
character has been sent to the user it is gone; there is no redacting it
afterwards. So the guard holds back the most recent `holdback_chars` and
emits only what lies behind that line. A pattern that fits inside the
hold-back is therefore always seen complete BEFORE any part of it is
emitted, and can be redacted properly.

The hold-back has a cost, paid in perceived latency: the user sees the
response trailing a few hundred characters behind the model. That is the
trade, and it is why the size is a knob rather than a constant.

What happens when a match is longer than the hold-back is the part worth
being explicit about, because it cannot be fixed by trying harder: its
first characters are already gone. The guard detects it anyway — that is
what the emitted tail in the scan window is for — and reports
`leaked_before_holdback=True`, which is a materially different incident
from a clean block and must not be logged as one. Set `holdback_chars`
above the longest credential your patterns can match and this does not
arise; the default of 256 clears every pattern shipped with the library.

--- 3. A verdict on a prefix is not a verdict ------------------------------

The system-prompt overlap score is a ratio over the whole response. Three
tokens into a stream that ratio is noise, and acting on it would block
responses for starting with a word that appears in the system prompt. So
overlap is not evaluated until `min_chars_for_overlap` have arrived. The
same caution does not apply to a credential match, which means exactly the
same thing on a prefix as on a whole.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exfil_guard import ExfilGuard, URLFinding
from .output_guard import (
    DEFAULT_MAX_SCAN_CHARS,
    OVERSIZED_OUTPUT_PLACEHOLDER,
    OutputGuard,
    OutputScanResult,
    _apply_spans,
    _Span,
    _tokenize,
    distinctive_tokens,
)

# Long enough for every secret pattern shipped in config/patterns.json,
# short enough that the lag is not felt as a stall. Raise it if you add a
# pattern that can match something longer.
DEFAULT_HOLDBACK_CHARS = 256

# Below this much text, the system-prompt overlap ratio is dominated by
# whichever common words happen to have arrived, so it is not consulted.
DEFAULT_MIN_CHARS_FOR_OVERLAP = 200

# How much already-emitted text stays in the scan window. This is NOT the
# hold-back and must not be tied to it: the hold-back decides what can
# still be redacted, this decides what can still be *seen*. Sizing it off
# the hold-back is a trap — lower the hold-back for latency and a long
# credential straddling the emission boundary stops being detected at all,
# rather than being detected and reported as a partial leak. It costs only
# memory, so it is generous by default.
DEFAULT_DETECTION_TAIL_CHARS = 512

BLOCKED_MESSAGE = (
    "[Response blocked by the security layer: possible credential "
    "or system-prompt leak detected.]"
)
EXFIL_BLOCKED_MESSAGE = (
    "[Response blocked by the security layer: it contained a URL that "
    "would have transmitted data to a third party when rendered.]"
)


@dataclass
class StreamDelta:
    """What the caller may forward, and whether to keep going.

    `text` is the safe, redacted text cleared for emission on this call. It
    is routinely empty: early chunks sit inside the hold-back window and
    nothing has cleared it yet.
    """

    text: str = ""
    blocked: bool = False
    reason: str | None = None
    # What to show instead of the whole response when blocked. The already
    # emitted text has to be replaced, not appended to.
    replacement_text: str | None = None
    # True when part of the offending match had already been emitted before
    # it could be recognized, i.e. it was longer than the hold-back window.
    # A different incident from a clean block: something did leak.
    leaked_before_holdback: bool = False
    secret_categories: tuple[str, ...] = ()
    pii_categories: tuple[str, ...] = ()
    exfil_reasons: tuple[str, ...] = ()
    system_prompt_overlap_score: float = 0.0

    @property
    def should_continue(self) -> bool:
        return not self.blocked


class StreamingOutputGuard:
    """Incremental counterpart to `OutputGuard.scan`.

        guard = StreamingOutputGuard(system_prompt=SYSTEM_PROMPT)
        async for chunk in model_stream:
            delta = guard.feed(chunk)
            if delta.blocked:
                replace_message(delta.replacement_text)
                break
            emit(delta.text)
        else:
            emit(guard.finish().text)

    Not thread-safe and not reentrant: one instance per response.
    """

    def __init__(
        self,
        output_guard: OutputGuard | None = None,
        exfil_guard: ExfilGuard | None = None,
        system_prompt: str | None = None,
        holdback_chars: int = DEFAULT_HOLDBACK_CHARS,
        overlap_threshold: float = 0.35,
        min_chars_for_overlap: int = DEFAULT_MIN_CHARS_FOR_OVERLAP,
        max_output_chars: int | None = DEFAULT_MAX_SCAN_CHARS,
        redact_pii: bool = True,
        detection_tail_chars: int | None = None,
    ):
        if holdback_chars < 0:
            raise ValueError("holdback_chars cannot be negative.")
        if detection_tail_chars is not None and detection_tail_chars < 0:
            raise ValueError("detection_tail_chars cannot be negative.")
        self.output_guard = output_guard or OutputGuard(max_scan_chars=None)
        self.exfil_guard = exfil_guard
        self.system_prompt = system_prompt
        self.holdback_chars = holdback_chars
        self.overlap_threshold = overlap_threshold
        self.min_chars_for_overlap = min_chars_for_overlap
        self.max_output_chars = max_output_chars
        self.redact_pii = redact_pii

        # Everything the model has produced, as produced. Kept for the final
        # scan and for the overlap ratio; bounded by max_output_chars.
        self._raw: list[str] = []
        self._raw_len = 0
        # Produced but not yet cleared for emission.
        self._buffer = ""
        # Never smaller than the hold-back: text still inside the window
        # has to remain visible to the scan after it is released.
        self._detection_tail_chars = max(
            holdback_chars,
            DEFAULT_DETECTION_TAIL_CHARS if detection_tail_chars is None else detection_tail_chars,
        )
        # The last `_detection_tail_chars` of raw text that WAS emitted.
        # Prefixed to the scan window so a match straddling the emission
        # boundary is still found, even though it can no longer be redacted.
        self._emitted_tail = ""
        self._emitted_len = 0
        # Tokens seen so far, for the overlap ratio without re-tokenizing
        # the whole response on every chunk.
        self._out_tokens: set[str] = set()
        # Distinctive tokens only, so the streamed score matches the
        # buffered one; see system_prompt_overlap.
        self._sys_tokens = distinctive_tokens(system_prompt) if system_prompt else set()

        self._finished = False
        self._blocked = False
        self._pii_categories: set[str] = set()
        self._secret_categories: set[str] = set()
        self._exfil_reasons: set[str] = set()
        self._overlap = 0.0
        self._leaked = False

    # -- public state ----------------------------------------------------

    @property
    def emitted_chars(self) -> int:
        """How much has already reached the user. Non-zero when a block
        arrives means the message needs replacing, not just truncating."""
        return self._emitted_len

    @property
    def raw_text(self) -> str:
        """Everything the model produced, unredacted."""
        return "".join(self._raw)

    @property
    def blocked(self) -> bool:
        return self._blocked

    # -- feeding ---------------------------------------------------------

    def feed(self, chunk: str) -> StreamDelta:
        if self._blocked:
            # Once blocked, stay blocked. A caller that keeps feeding gets
            # the same refusal rather than a window that reopens.
            return self._block_delta(self._reason or "blocked", replacement=self._replacement)
        if self._finished:
            raise RuntimeError("StreamingOutputGuard.feed() called after finish().")
        if not chunk:
            return StreamDelta(system_prompt_overlap_score=self._overlap)

        self._raw.append(chunk)
        self._raw_len += len(chunk)
        self._buffer += chunk
        self._absorb_tokens(chunk)

        if self.max_output_chars is not None and self._raw_len > self.max_output_chars:
            # Same reasoning as the non-streaming size cap: text that was
            # never scanned must not be forwarded as though it had been.
            return self._block(
                "oversized", replacement=OVERSIZED_OUTPUT_PLACEHOLDER,
            )

        return self._evaluate(final=False)

    def finish(self) -> StreamDelta:
        """Flush the hold-back and apply the checks that need the whole text.

        Must be called: whatever is inside the hold-back window when the
        model stops has not been emitted yet, and the overlap ratio has
        possibly never been evaluated on a stream shorter than
        `min_chars_for_overlap`.
        """
        if self._blocked:
            return self._block_delta(self._reason or "blocked", replacement=self._replacement)
        self._finished = True
        return self._evaluate(final=True)

    def result(self) -> OutputScanResult:
        """A scan result over the whole response, for the audit record.

        The same shape `post_process` produces, so a streamed response and a
        buffered one log identically.
        """
        raw = self.raw_text
        return OutputScanResult(
            original_text=raw,
            redacted_text=self._replacement if self._blocked else raw,
            secret_findings={c: [] for c in sorted(self._secret_categories)},
            pii_findings={c: [] for c in sorted(self._pii_categories)},
            system_prompt_overlap_score=self._overlap,
            blocked=self._blocked,
        )

    # -- internals -------------------------------------------------------

    _reason: str | None = None
    _replacement: str = BLOCKED_MESSAGE

    def _absorb_tokens(self, chunk: str) -> None:
        if not self._sys_tokens:
            return
        # Re-tokenize a small tail together with the chunk so a word split
        # across the boundary ("sys" + "tem") is not counted as two.
        tail = self._buffer[-len(chunk) - 32:] if len(self._buffer) > len(chunk) else chunk
        self._out_tokens |= _tokenize(tail)

    def _overlap_score(self) -> float:
        if not self._sys_tokens:
            return 0.0
        return round(len(self._sys_tokens & self._out_tokens) / len(self._sys_tokens), 3)

    def _evaluate(self, final: bool) -> StreamDelta:
        # The window is the unemitted buffer with the tail of what was
        # already emitted glued to the front, so a match spanning the seam
        # is found. `offset` is where the unemitted part begins.
        offset = len(self._emitted_tail)
        window = self._emitted_tail + self._buffer

        secret_spans = self.output_guard._secret_spans(window)
        pii_spans = self.output_guard._pii_spans(window) if self.redact_pii else []
        url_findings = self.exfil_guard.find_urls(window) if self.exfil_guard is not None else []

        # Anything that ends inside the already-emitted tail was dealt with
        # on an earlier pass; re-reporting it would double-count.
        secret_spans = [s for s in secret_spans if s.end > offset]
        pii_spans = [s for s in pii_spans if s.end > offset]
        url_findings = [f for f in url_findings if f.span[1] > offset]

        self._pii_categories.update(s.category for s in pii_spans)

        if secret_spans:
            self._secret_categories.update(s.category for s in secret_spans)
            straddles = any(s.start < offset for s in secret_spans)
            return self._block("secret", straddles=straddles)

        blocking_urls = [f for f in url_findings if f.severity == "block"]
        if blocking_urls:
            self._exfil_reasons.update(r for f in blocking_urls for r in f.reasons)
            return self._block(
                "exfil",
                replacement=EXFIL_BLOCKED_MESSAGE,
                straddles=any(f.span[0] < offset for f in blocking_urls),
            )

        self._overlap = self._overlap_score()
        if (
            self._sys_tokens
            and (final or self._raw_len >= self.min_chars_for_overlap)
            and self._overlap >= self.overlap_threshold
        ):
            return self._block("system_prompt_overlap")

        return self._emit(window, offset, pii_spans, url_findings, final)

    def _emit(
        self,
        window: str,
        offset: int,
        pii_spans: list[_Span],
        url_findings: list[URLFinding],
        final: bool,
    ) -> StreamDelta:
        # Where the emission boundary falls in window coordinates. On the
        # final call the hold-back is released: there is no more text
        # coming, so nothing can grow into a match any more.
        cut = len(window) if final else max(offset, len(window) - self.holdback_chars)

        # Never cut through the middle of a finding: half of a redaction is
        # a leak of the other half.
        for start, end in [(s.start, s.end) for s in pii_spans] + [f.span for f in url_findings]:
            if start < cut < end:
                cut = start

        if cut <= offset:
            return StreamDelta(system_prompt_overlap_score=self._overlap)

        emitted_window = window[offset:cut]
        spans_in_emitted = [
            _Span(s.start - offset, s.end - offset, s.category, s.priority)
            for s in pii_spans
            if s.end <= cut
        ]
        text = _apply_spans(emitted_window, spans_in_emitted)

        if self.exfil_guard is not None:
            neutralizable = [f for f in url_findings if f.suspicious and f.span[1] <= cut]
            if neutralizable:
                # Re-scan the cleared slice rather than reusing window
                # offsets: neutralization rewrites the text, and mapping
                # spans across a rewrite that has already happened once
                # (the PII redaction above) is how off-by-one leaks happen.
                text = self.exfil_guard.scan(text).neutralized_text

        # A finding that starts before the emission boundary was already
        # partly sent: it is redacted from here on, but the front of it is
        # gone. Same limit as a straddling secret, and reported the same
        # way rather than quietly producing a half-redacted email.
        straddled = any(s.start < offset for s in pii_spans) or any(
            f.span[0] < offset for f in url_findings if f.suspicious
        )
        if straddled and self._emitted_len > 0:
            self._leaked = True

        self._advance(window[offset:cut])
        return StreamDelta(
            text=text,
            pii_categories=tuple(sorted(self._pii_categories)),
            system_prompt_overlap_score=self._overlap,
            leaked_before_holdback=self._leaked,
        )

    def _advance(self, emitted_raw: str) -> None:
        self._buffer = self._buffer[len(emitted_raw):]
        self._emitted_len += len(emitted_raw)
        tail = self._emitted_tail + emitted_raw
        self._emitted_tail = tail[-self._detection_tail_chars:] if self._detection_tail_chars else ""

    def _block(
        self, reason: str, replacement: str = BLOCKED_MESSAGE, straddles: bool = False,
    ) -> StreamDelta:
        self._blocked = True
        self._reason = reason
        self._replacement = replacement
        # Straddling only matters if something was actually emitted; on the
        # first chunk the tail is empty and nothing has leaked.
        self._leaked = straddles and self._emitted_len > 0
        self._buffer = ""
        return self._block_delta(reason, replacement)

    def _block_delta(self, reason: str, replacement: str) -> StreamDelta:
        return StreamDelta(
            text="",
            blocked=True,
            reason=reason,
            replacement_text=replacement,
            leaked_before_holdback=self._leaked,
            secret_categories=tuple(sorted(self._secret_categories)),
            pii_categories=tuple(sorted(self._pii_categories)),
            exfil_reasons=tuple(sorted(self._exfil_reasons)),
            system_prompt_overlap_score=self._overlap,
        )
