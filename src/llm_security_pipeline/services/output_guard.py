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
from dataclasses import dataclass, field

from ..config_loader import PatternRegistry

# ---------------------------------------------------------------------------
# Patterns requiring special handling beyond a plain regex match
# ---------------------------------------------------------------------------

_CC_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}")


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


def system_prompt_overlap(output_text: str, system_prompt: str) -> float:
    """Return the fraction of system-prompt tokens that also appear in the
    output. A high value is a strong signal of leakage, even partial or
    translated (proper nouns, technical terms and config IDs tend to stay
    identical even across a translation)."""
    sys_tokens = _tokenize(system_prompt)
    out_tokens = _tokenize(output_text)
    if not sys_tokens:
        return 0.0
    overlap = len(sys_tokens & out_tokens) / len(sys_tokens)
    return round(overlap, 3)


@dataclass
class OutputScanResult:
    original_text: str
    redacted_text: str
    secret_findings: dict[str, list[str]] = field(default_factory=dict)
    pii_findings: dict[str, list[str]] = field(default_factory=dict)
    system_prompt_overlap_score: float = 0.0
    blocked: bool = False


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
    ):
        self.registry = registry or PatternRegistry.load(
            custom_config_path=pattern_config_path,
            include_defaults=include_default_patterns,
        )

    def find_secrets(self, text: str) -> dict[str, list[str]]:
        findings: dict[str, list[str]] = {}
        for name, pattern in self.registry.secret_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # findall() returns tuples when the pattern has capture
                # groups; normalize to the full match text in that case.
                normalized_matches = [
                    m if isinstance(m, str) else pattern.search(text).group(0)
                    for m in matches
                ]
                findings[name] = normalized_matches
        return findings

    def find_pii(self, text: str) -> dict[str, list[str]]:
        findings: dict[str, list[str]] = {}

        for name, pattern in self.registry.pii_patterns.items():
            matches = [m.group(0) for m in pattern.finditer(text)]
            if matches:
                findings[name] = matches

        # Credit card: regex only finds *candidates*, Luhn checksum decides.
        cards = [m for m in _CC_CANDIDATE_RE.findall(text) if _luhn_check(m)]
        if cards:
            findings["credit_card"] = cards

        # Phone numbers: the regex is intentionally permissive to cover many
        # international formats, which means it produces false positives.
        # Treat this category as a signal to review, not an automatic block.
        phones = [
            m.group(0).strip()
            for m in _PHONE_RE.finditer(text)
            if len(re.sub(r"\D", "", m.group(0))) >= 8
        ]
        if phones:
            findings["phone_candidate"] = phones

        return findings

    @staticmethod
    def redact(text: str, findings: dict[str, list[str]]) -> str:
        redacted = text
        for category, matches in findings.items():
            for m in matches:
                if m:
                    redacted = redacted.replace(m, f"[REDACTED:{category.upper()}]")
        return redacted

    def scan(
        self,
        text: str,
        system_prompt: str | None = None,
        overlap_threshold: float = 0.35,
    ) -> OutputScanResult:
        secrets_found = self.find_secrets(text)
        pii_found = self.find_pii(text)

        combined = {**secrets_found, **pii_found}
        redacted = self.redact(text, combined)

        overlap = 0.0
        if system_prompt:
            overlap = system_prompt_overlap(text, system_prompt)

        blocked = bool(secrets_found) or overlap >= overlap_threshold

        return OutputScanResult(
            original_text=text,
            redacted_text=redacted,
            secret_findings=secrets_found,
            pii_findings=pii_found,
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
