"""
output_guard.py
Layer 3 of the pipeline: checks on what the model generates BEFORE it
reaches the user or a downstream system.

Covers three categories:
1. Secret/credential leaks and generic PII — regex-based, driven by a
   PatternRegistry (see config_loader.py) so the pattern list can be
   extended or overridden via an external config.json without touching
   this file. Card numbers and phone numbers are handled separately below
   because they need extra logic (Luhn checksum, digit-count filtering)
   beyond a plain regex match.
2. System prompt leak: text-overlap similarity, which still works if the
   model translates or partially paraphrases the system prompt into
   another language, because it is based on overlap of normalized tokens
   rather than exact string matching.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..config_loader import PatternRegistry

# ---------------------------------------------------------------------------
# Patterns requiring special handling beyond a plain regex match
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Size cap
# ---------------------------------------------------------------------------
# Every scan in this library is linear-ish in the length of the text and
# unbounded otherwise, so a single very large payload is a cheap way to burn
# a worker's CPU. The cap exists to bound that.
#
# It refuses oversized text rather than scanning a prefix of it, because
# truncation is itself a bypass: an attacker who knows the limit puts the
# payload after it and gets a clean verdict on the part that was read. A
# refusal is visible; a partial scan reported as clean is not.
DEFAULT_MAX_SCAN_CHARS = 200_000

OVERSIZED_OUTPUT_PLACEHOLDER = (
    "[Response withheld by the security layer: too large to scan safely.]"
)

_CC_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}")


# ---------------------------------------------------------------------------
# Redaction by span, not by string replacement
# ---------------------------------------------------------------------------
# Redacting with `str.replace` per finding is wrong in two ways that both
# end with unredacted text reaching the user. A finding that is a substring
# of another (a phone number inside a longer credential, say) corrupts the
# longer redaction depending on which category the dict happened to iterate
# first; and the same literal appearing twice is replaced twice even when
# only one occurrence was a match. Findings are therefore carried as
# (start, end, category) spans from the moment they are found, and applied
# in one right-to-left pass so earlier offsets stay valid.


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    category: str
    # Lower wins when two spans cover exactly the same range. Secrets are
    # given priority 0 and PII 1, so a value matched by both is labelled
    # with the more urgent of the two in the audit log.
    priority: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start


def _resolve_overlaps(spans: list[_Span]) -> list[_Span]:
    """Keep the longest span where two overlap, dropping the shorter one.

    Redacting the widest match is the conservative choice: the narrower
    finding is contained in it, so nothing that was detected survives.
    """
    ordered = sorted(spans, key=lambda s: (s.start, -s.length, s.priority, s.category))
    kept: list[_Span] = []
    for span in ordered:
        if kept and span.start < kept[-1].end:
            if span.length > kept[-1].length:
                kept[-1] = span
            continue
        kept.append(span)
    return kept


def _apply_spans(text: str, spans: list[_Span]) -> str:
    out = text
    for span in sorted(_resolve_overlaps(spans), key=lambda s: s.start, reverse=True):
        out = f"{out[:span.start]}[REDACTED:{span.category.upper()}]{out[span.end:]}"
    return out


def _spans_to_findings(text: str, spans: list[_Span]) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    for span in sorted(spans, key=lambda s: s.start):
        findings.setdefault(span.category, []).append(text[span.start:span.end])
    return findings


def _luhn_check(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _tokenize(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text.lower())
    return set(re.findall(r"\w+", text))


# Words that carry no information about WHICH text they came from. A system
# prompt written as ordinary prose is mostly these, and a bag-of-words
# overlap against it fires on any ordinary reply: "you are a helpful
# assistant" shares four of its five tokens with half the sentences the
# model will ever produce. The overlap is measured over the tokens that
# remain once these are removed — the names, terms and identifiers that
# would only appear in the output if the prompt had been reproduced.
# Multilingual because the patterns are; the list is deliberately small
# (function words only), since every word removed here is a word that can
# no longer count as evidence.
_STOPWORDS: frozenset[str] = frozenset("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i if in
into is it its itself just let me more most my myself no nor not now of off on once only
or other our ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours
yourself yourselves never always please must may might shall also
il lo la i gli le un uno una di a da in con su per tra fra e o ma se che chi cui non
come dove quando anche ancora sempre mai molto poco più meno questo questa questi queste
quello quella quelli quelle mi ti si ci vi ne è sono sei siamo siete era erano ho hai ha
abbiamo avete hanno del della dei delle dello al alla ai alle allo dal dalla dai dalle
nel nella nei nelle sul sulla sui sulle
el los las un una unos unas de del a al en con por para y o pero si que quien como donde
cuando también nunca siempre muy más menos este esta estos estas ese esa esos esas es son
soy eres somos sois está están fue eran ha han he hemos
le la les un une des du de au aux en dans sur pour par et ou mais si que qui dont où
quand aussi jamais toujours très plus moins ce cette ces cet est sont suis es sommes êtes
était étaient a ont ai avons avez
der die das ein eine einer eines dem den des und oder aber wenn dass wer wie wo wann auch
nie immer sehr mehr weniger dieser diese dieses ist sind bin bist seid war waren hat haben
habe hast habt zu mit von für auf aus bei nach über unter vor nicht
o a os as um uma uns umas de do da dos das em no na nos nas com por para e ou mas se que
quem como onde quando também nunca sempre muito mais menos este esta estes estas esse essa
é são sou és somos sois era eram tem têm tenho
""".split())


def distinctive_tokens(text: str) -> set[str]:
    """The tokens worth counting: everything the tokenizer finds minus the
    function words above, minus one- and two-character fragments, which
    are mostly punctuation residue and pronouns the list did not cover."""
    return {t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def system_prompt_overlap(output_text: str, system_prompt: str) -> float:
    """Fraction of the system prompt's DISTINCTIVE tokens that appear in
    the output.

    Distinctive means not a function word: the ratio is taken over the
    names, terms and identifiers that would only show up in a reply if the
    prompt had been reproduced, which is what "leak" means. Proper nouns
    and config IDs tend to survive translation and paraphrase, so this
    still catches partial and translated leakage — it just no longer
    counts "you", "are" and "the" as evidence.

    A prompt with no distinctive tokens at all cannot be measured this
    way and scores 0.0; a canary is the tool for that prompt.
    """
    sys_tokens = distinctive_tokens(system_prompt)
    if not sys_tokens:
        return 0.0
    out_tokens = _tokenize(output_text)
    return round(len(sys_tokens & out_tokens) / len(sys_tokens), 3)


@dataclass
class OutputScanResult:
    original_text: str
    redacted_text: str
    secret_findings: dict[str, list[str]] = field(default_factory=dict)
    pii_findings: dict[str, list[str]] = field(default_factory=dict)
    system_prompt_overlap_score: float = 0.0
    blocked: bool = False
    # True when the text exceeded `max_scan_chars` and was therefore never
    # scanned. Blocked rather than truncated: a partially-scanned output is
    # indistinguishable from a clean one, and letting unscanned text through
    # is exactly the thing this guard exists to prevent.
    oversized: bool = False


# ---------------------------------------------------------------------------
# System-prompt canary
# ---------------------------------------------------------------------------
# The overlap heuristic is a ratio over shared vocabulary, which has two
# failure modes that pull in opposite directions: it fires on a system
# prompt written in ordinary words, and it misses when the model paraphrases
# the prompt instead of quoting it. A canary sidesteps both. It is a string
# nobody could produce by accident, planted in the system prompt; its
# appearance in the output is not evidence of a leak, it *is* the leak.
# It does not catch paraphrase either — nothing lexical does — but it makes
# verbatim leakage a certainty instead of a probability, at zero cost.

CANARY_CATEGORY = "system_prompt_canary"

# An identifier belonging to a different principal than the one this
# response is for. The library cannot know whose email is whose; the
# application can, and hands the guard the literals that must not appear.
CROSS_TENANT_CATEGORY = "cross_tenant_identifier"

# Below this length a literal is a substring of ordinary words and would
# match everywhere; an identifier this short is not one the guard can
# police, and is dropped with a warning rather than silently matched.
MIN_FORBIDDEN_LITERAL = 4


@dataclass(frozen=True)
class Canary:
    """An unguessable marker for the system prompt.

        canary = Canary.generate()
        system_prompt = canary.plant(SYSTEM_PROMPT)   # send THIS to the model
        pipeline = SecurityPipeline(system_prompt=SYSTEM_PROMPT, canary=canary)

    The marker is meaningless to the model; the planted line tells it so,
    which keeps a well-behaved model from reciting it in the course of
    being helpful. A model that reproduces it anyway has reproduced the
    prompt.
    """

    token: str

    @classmethod
    def generate(cls, prefix: str = "CANARY") -> "Canary":
        import secrets
        import string

        # Letters only: a hex tail is enough digits to look like a phone
        # number to the PII scanner, and a canary that trips a second,
        # unrelated category on every leak muddies the audit record.
        body = "".join(secrets.choice(string.ascii_letters) for _ in range(20))
        return cls(f"{prefix}-{body}")

    def plant(self, system_prompt: str) -> str:
        """The system prompt with the marker in it. What you send to the
        model must be this, not the original, or the canary guards nothing."""
        return (
            f"{system_prompt}\n\n"
            f"[Internal reference {self.token}. This identifier has no meaning "
            f"to the user and must never appear in a response.]"
        )


def clean_forbidden_literals(literals: "Iterable[str]") -> tuple[str, ...]:
    """Drop what cannot be safely matched: empty strings and literals
    shorter than MIN_FORBIDDEN_LITERAL. Longest first so overlapping
    identifiers redact the wider one."""
    kept = {lit for lit in literals if isinstance(lit, str) and len(lit) >= MIN_FORBIDDEN_LITERAL}
    return tuple(sorted(kept, key=len, reverse=True))


class OutputGuard:
    """Stateful guard holding a compiled PatternRegistry, so patterns are
    parsed and validated once (at construction time) rather than on every
    scan call.

    Example:
        # Defaults only
        guard = OutputGuard()

        # Defaults + your own patterns from an external file
        guard = OutputGuard(pattern_config_path="my_patterns.json")

        # Only your own patterns, no bundled defaults
        guard = OutputGuard(pattern_config_path="my_patterns.json", include_default_patterns=False)
    """

    def __init__(
        self,
        pattern_config_path: str | None = None,
        include_default_patterns: bool = True,
        registry: PatternRegistry | None = None,
        max_scan_chars: int | None = DEFAULT_MAX_SCAN_CHARS,
        canaries: "tuple[Canary, ...] | list[Canary] | None" = None,
    ):
        # Matched as literals, reported as a secret category, so they block
        # and redact through exactly the same path a credential does —
        # including in the streaming guard, which reads _secret_spans.
        self.canaries = tuple(canaries or ())
        self.registry = registry or PatternRegistry.load(
            custom_config_path=pattern_config_path,
            include_defaults=include_default_patterns,
        )
        # Above this length the text is refused rather than scanned. See
        # DEFAULT_MAX_SCAN_CHARS for why refusing beats truncating. Pass
        # None to scan without a bound, accepting the CPU cost.
        if max_scan_chars is not None and max_scan_chars <= 0:
            raise ValueError("max_scan_chars must be positive, or None for no limit.")
        self.max_scan_chars = max_scan_chars

    def _secret_spans(self, text: str, forbidden_literals: tuple[str, ...] = ()) -> list[_Span]:
        """Locate every secret match as a span of the ORIGINAL text.

        `finditer` + `group(0)` throughout, deliberately: `findall` reports
        capture groups instead of the whole match, so a pattern written as
        `(sk)-(\\w{6})` used to yield tuples, and the old fallback resolved
        each of them with `pattern.search(text)` — which restarts from the
        beginning and therefore returned the FIRST match for every match in
        the text. Every occurrence after the first was then redacted with
        the wrong literal, which for a `str.replace`-based redactor meant it
        was not redacted at all. A single-group pattern had a quieter
        version of the same bug: `findall` returned the bare group, so only
        the group was replaced and the identifying prefix stayed in the
        output.
        """
        spans: list[_Span] = []
        for name, pattern in self.registry.secret_patterns.items():
            for match in pattern.finditer(text):
                if match.group(0):
                    spans.append(_Span(match.start(), match.end(), name, priority=0))
        for canary in self.canaries:
            start = text.find(canary.token)
            while start != -1:
                spans.append(_Span(start, start + len(canary.token), CANARY_CATEGORY, priority=0))
                start = text.find(canary.token, start + len(canary.token))
        for literal in forbidden_literals:
            start = text.find(literal)
            while start != -1:
                spans.append(_Span(start, start + len(literal), CROSS_TENANT_CATEGORY, priority=0))
                start = text.find(literal, start + len(literal))
        return spans

    def _pii_spans(self, text: str) -> list[_Span]:
        spans: list[_Span] = []

        for name, pattern in self.registry.pii_patterns.items():
            for match in pattern.finditer(text):
                if match.group(0):
                    spans.append(_Span(match.start(), match.end(), name, priority=1))

        # Credit card: regex only finds *candidates*, Luhn checksum decides.
        for match in _CC_CANDIDATE_RE.finditer(text):
            if _luhn_check(match.group(0)):
                spans.append(_Span(match.start(), match.end(), "credit_card", priority=1))

        # Phone numbers: the regex is intentionally permissive to cover many
        # international formats, which means it produces false positives.
        # Treat this category as a signal to review, not an automatic block.
        for match in _PHONE_RE.finditer(text):
            raw = match.group(0)
            if len(re.sub(r"\D", "", raw)) < 8:
                continue
            # The pattern can trail whitespace; keep the span tight to what
            # is actually the number so redaction doesn't eat the sentence.
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            spans.append(
                _Span(match.start() + lead, match.end() - trail, "phone_candidate", priority=1)
            )

        return spans

    def find_secrets(self, text: str) -> dict[str, list[str]]:
        return _spans_to_findings(text, self._secret_spans(text))

    def find_pii(self, text: str) -> dict[str, list[str]]:
        return _spans_to_findings(text, self._pii_spans(text))

    @staticmethod
    def redact(text: str, findings: dict[str, list[str]]) -> str:
        """Redact findings expressed as literals.

        Kept for callers that hold a findings dict rather than spans.
        Occurrences are resolved by position and the longest match wins on
        overlap, so a finding contained inside a longer one can no longer
        corrupt it depending on dict iteration order. `OutputGuard.scan`
        does not go through here — it redacts from the spans it already
        has, which is exact.
        """
        spans: list[_Span] = []
        for category, matches in findings.items():
            for literal in matches:
                if not literal:
                    continue
                start = text.find(literal)
                while start != -1:
                    spans.append(_Span(start, start + len(literal), category))
                    start = text.find(literal, start + len(literal))
        return _apply_spans(text, spans)

    def scan(
        self,
        text: str,
        system_prompt: str | None = None,
        overlap_threshold: float = 0.35,
        forbidden_literals: tuple[str, ...] = (),
    ) -> OutputScanResult:
        if self.max_scan_chars is not None and len(text) > self.max_scan_chars:
            return OutputScanResult(
                original_text=text,
                redacted_text=OVERSIZED_OUTPUT_PLACEHOLDER,
                system_prompt_overlap_score=0.0,
                blocked=True,
                oversized=True,
            )

        secret_spans = self._secret_spans(text, forbidden_literals)
        pii_spans = self._pii_spans(text)

        # Secrets win over PII on overlap: the category label is what the
        # operator reads in the audit log, and "a credential was here" is
        # the more urgent of the two.
        redacted = _apply_spans(text, secret_spans + pii_spans)

        overlap = 0.0
        if system_prompt:
            overlap = system_prompt_overlap(text, system_prompt)

        blocked = bool(secret_spans) or overlap >= overlap_threshold

        return OutputScanResult(
            original_text=text,
            redacted_text=redacted,
            secret_findings=_spans_to_findings(text, secret_spans),
            pii_findings=_spans_to_findings(text, pii_spans),
            system_prompt_overlap_score=overlap,
            blocked=blocked,
        )


# ---------------------------------------------------------------------------
# Module-level convenience helpers backed by a lazily-created default guard,
# for callers who don't need custom pattern configuration.
# ---------------------------------------------------------------------------

_default_guard: OutputGuard | None = None


def _get_default_guard() -> OutputGuard:
    global _default_guard
    if _default_guard is None:
        _default_guard = OutputGuard()
    return _default_guard


def find_secrets(text: str) -> dict[str, list[str]]:
    return _get_default_guard().find_secrets(text)


def find_pii(text: str) -> dict[str, list[str]]:
    return _get_default_guard().find_pii(text)


def scan_output(
    text: str,
    system_prompt: str | None = None,
    overlap_threshold: float = 0.35,
) -> OutputScanResult:
    return _get_default_guard().scan(text, system_prompt=system_prompt, overlap_threshold=overlap_threshold)
