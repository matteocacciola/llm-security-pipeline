"""
config_loader.py
Loads regex-based detection patterns (secrets, PII, multilingual injection
phrases) from external JSON config files, so pattern lists can be extended
or overridden without touching pipeline source code.

Expected JSON shape (see config/patterns.json for the bundled defaults):

{
  "secret_patterns": [
    {"name": "openai_api_key", "pattern": "sk-[A-Za-z0-9]{20,}", "flags": ["IGNORECASE"]}
  ],
  "pii_patterns": [
    {"name": "it_codice_fiscale", "pattern": "...", "flags": []}
  ],
  "injection_patterns": [
    {"name": "en_ignore_previous_instructions", "lang": "en", "pattern": "...", "flags": ["IGNORECASE"]}
  ]
}

`injection_patterns` entries carry an extra required `lang` field (the
heuristic is grouped by language so callers can tell which languages a
piece of text tripped, e.g. for a code-mixing signal) but are otherwise
the same shape as secret/PII entries.

Every entry is validated against a pydantic schema (required fields,
known flag names, syntactically valid regex) before it ever reaches the
sanitizer/output guard, so a malformed config fails loudly and precisely
at load time — with a field-level error pointing at exactly which entry
is wrong — rather than as a confusing error deep in the scanning path, or
worse, a silently-skipped pattern.

Usage:
    from config_loader import PatternRegistry

    # Defaults only
    registry = PatternRegistry.load()

    # Defaults + your own additions/overrides
    registry = PatternRegistry.load(custom_config_path="my_patterns.json")

    # Only your own patterns, ignore bundled defaults entirely
    registry = PatternRegistry.load(custom_config_path="my_patterns.json", include_defaults=False)

Entries in a custom file with the same "name" as a default entry REPLACE
that default; entries with a new name are added alongside the defaults.
This applies independently within each of the three categories.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "patterns.json"

_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}

_SECRET_CATEGORY = "secret_patterns"
_PII_CATEGORY = "pii_patterns"
_INJECTION_CATEGORY = "injection_patterns"


class PatternConfigError(Exception):
    """Raised when a pattern config file is missing, malformed, or contains
    an invalid regular expression."""


# ---------------------------------------------------------------------------
# Schema (pydantic): validates the raw JSON structure before anything is
# compiled or handed to the sanitizer/output guard.
# ---------------------------------------------------------------------------

class _PatternEntry(BaseModel):
    """Schema for one secret_patterns/pii_patterns entry."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    flags: list[str] = Field(default_factory=list)

    @field_validator("flags")
    @classmethod
    def _flags_are_known(cls, value: list[str]) -> list[str]:
        for flag_name in value:
            if str(flag_name).upper() not in _FLAG_MAP:
                raise ValueError(f"Unknown regex flag '{flag_name}'. Supported flags: {list(_FLAG_MAP)}")
        return value

    @field_validator("pattern")
    @classmethod
    def _pattern_is_valid_regex(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc
        return value

    def resolved_flags(self) -> int:
        flags = 0
        for flag_name in self.flags:
            flags |= _FLAG_MAP[flag_name.upper()]
        return flags


class _InjectionPatternEntry(_PatternEntry):
    """Same shape as _PatternEntry, plus the 'lang' field the multilingual
    injection heuristic groups by."""

    lang: str = Field(min_length=1)


class _PatternConfigFile(BaseModel):
    """Top-level schema for a whole patterns.json file."""

    model_config = ConfigDict(extra="ignore")  # tolerate "_comment" and similar metadata keys

    secret_patterns: list[_PatternEntry] = Field(default_factory=list)
    pii_patterns: list[_PatternEntry] = Field(default_factory=list)
    injection_patterns: list[_InjectionPatternEntry] = Field(default_factory=list)

    @field_validator("secret_patterns", "pii_patterns", "injection_patterns")
    @classmethod
    def _names_are_unique(cls, entries: list) -> list:
        # Patterns are keyed by name once loaded, so a duplicate would
        # silently replace the one before it: a rule someone wrote and
        # believes is active, and is not. Refused at load instead.
        seen: set[str] = set()
        for entry in entries:
            if entry.name in seen:
                raise ValueError(f"duplicate pattern name {entry.name!r} in the same category")
            seen.add(entry.name)
        return entries


def load_pattern_file(path: str | Path) -> dict[str, dict]:
    """Load, validate and compile a single pattern config file. Raises
    PatternConfigError on any structural, schema or regex problem, so
    misconfiguration fails loudly at startup rather than silently
    disabling detection at runtime."""
    path = Path(path)
    if not path.exists():
        raise PatternConfigError(f"Pattern config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise PatternConfigError(f"Malformed JSON in pattern config file {path}: {exc}") from exc

    try:
        config = _PatternConfigFile.model_validate(raw)
    except ValidationError as exc:
        raise PatternConfigError(f"Invalid pattern config file {path}:\n{exc}") from exc

    return {
        _SECRET_CATEGORY: {e.name: re.compile(e.pattern, e.resolved_flags()) for e in config.secret_patterns},
        _PII_CATEGORY: {e.name: re.compile(e.pattern, e.resolved_flags()) for e in config.pii_patterns},
        _INJECTION_CATEGORY: {
            e.name: (e.lang, re.compile(e.pattern, e.resolved_flags())) for e in config.injection_patterns
        },
    }


@dataclass
class PatternRegistry:
    """Holds the merged, compiled set of secret, PII and multilingual
    injection-phrase detection patterns used by output_guard.py and
    sanitizer.py respectively."""

    secret_patterns: dict[str, re.Pattern]
    pii_patterns: dict[str, re.Pattern]
    injection_patterns: dict[str, dict[str, re.Pattern]]  # lang -> {name: pattern}

    @classmethod
    def load(
        cls,
        custom_config_path: str | Path | None = None,
        include_defaults: bool = True,
    ) -> "PatternRegistry":
        secret_patterns: dict[str, re.Pattern] = {}
        pii_patterns: dict[str, re.Pattern] = {}
        injection_by_name: dict[str, tuple[str, re.Pattern]] = {}

        if include_defaults:
            defaults = load_pattern_file(DEFAULT_CONFIG_PATH)
            secret_patterns.update(defaults[_SECRET_CATEGORY])
            pii_patterns.update(defaults[_PII_CATEGORY])
            injection_by_name.update(defaults[_INJECTION_CATEGORY])

        if custom_config_path:
            custom = load_pattern_file(custom_config_path)
            # Custom entries override defaults sharing the same name, and are
            # added alongside the rest.
            secret_patterns.update(custom[_SECRET_CATEGORY])
            pii_patterns.update(custom[_PII_CATEGORY])
            injection_by_name.update(custom[_INJECTION_CATEGORY])

        if not secret_patterns and not pii_patterns and not injection_by_name:
            raise PatternConfigError(
                "PatternRegistry loaded with zero patterns. Check that "
                "include_defaults=True or that custom_config_path points to a valid file."
            )

        injection_patterns: dict[str, dict[str, re.Pattern]] = {}
        for name, (lang, pattern) in injection_by_name.items():
            injection_patterns.setdefault(lang, {})[name] = pattern

        return cls(
            secret_patterns=secret_patterns,
            pii_patterns=pii_patterns,
            injection_patterns=injection_patterns,
        )