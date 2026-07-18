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

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "patterns.json"

_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}

_VALID_CATEGORIES = ("secret_patterns", "pii_patterns")
_INJECTION_CATEGORY = "injection_patterns"


class PatternConfigError(Exception):
    """Raised when a pattern config file is missing, malformed, or contains
    an invalid regular expression."""


def _resolve_flags(entry: dict, name: str, category_label: str) -> int:
    flags = 0
    for flag_name in entry.get("flags", []):
        resolved = _FLAG_MAP.get(str(flag_name).upper())
        if resolved is None:
            raise PatternConfigError(
                f"Unknown regex flag '{flag_name}' for pattern '{name}' in '{category_label}'. "
                f"Supported flags: {list(_FLAG_MAP)}"
            )
        flags |= resolved
    return flags


def _compile_pattern_list(raw_list: list[dict], category_label: str) -> dict[str, re.Pattern]:
    compiled: dict[str, re.Pattern] = {}
    for entry in raw_list:
        name = entry.get("name")
        pattern = entry.get("pattern")
        if not name or not pattern:
            raise PatternConfigError(
                f"Invalid pattern entry in '{category_label}' (missing 'name' or 'pattern'): {entry}"
            )
        flags = _resolve_flags(entry, name, category_label)
        try:
            compiled[name] = re.compile(pattern, flags)
        except re.error as exc:
            raise PatternConfigError(
                f"Invalid regular expression for pattern '{name}' in '{category_label}': {exc}"
            ) from exc
    return compiled


def _compile_injection_pattern_list(raw_list: list[dict], category_label: str) -> dict[str, tuple[str, re.Pattern]]:
    """Same shape as _compile_pattern_list, plus the required 'lang' field.
    Returns name -> (lang, compiled_pattern) so callers can regroup by
    language while still merging/overriding by the same 'name' key as the
    other two categories."""
    compiled: dict[str, tuple[str, re.Pattern]] = {}
    for entry in raw_list:
        name = entry.get("name")
        lang = entry.get("lang")
        pattern = entry.get("pattern")
        if not name or not lang or not pattern:
            raise PatternConfigError(
                f"Invalid pattern entry in '{category_label}' (missing 'name', 'lang' or 'pattern'): {entry}"
            )
        flags = _resolve_flags(entry, name, category_label)
        try:
            compiled[name] = (lang, re.compile(pattern, flags))
        except re.error as exc:
            raise PatternConfigError(
                f"Invalid regular expression for pattern '{name}' in '{category_label}': {exc}"
            ) from exc
    return compiled


def load_pattern_file(path: str | Path) -> dict[str, dict]:
    """Load and compile a single pattern config file. Raises PatternConfigError
    on any structural or regex problem, so misconfiguration fails loudly at
    startup rather than silently disabling detection at runtime."""
    path = Path(path)
    if not path.exists():
        raise PatternConfigError(f"Pattern config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise PatternConfigError(f"Malformed JSON in pattern config file {path}: {exc}") from exc

    result: dict[str, dict] = {}
    for category in _VALID_CATEGORIES:
        result[category] = _compile_pattern_list(raw.get(category, []), category)
    result[_INJECTION_CATEGORY] = _compile_injection_pattern_list(
        raw.get(_INJECTION_CATEGORY, []), _INJECTION_CATEGORY
    )
    return result


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
            secret_patterns.update(defaults["secret_patterns"])
            pii_patterns.update(defaults["pii_patterns"])
            injection_by_name.update(defaults[_INJECTION_CATEGORY])

        if custom_config_path:
            custom = load_pattern_file(custom_config_path)
            # Custom entries override defaults sharing the same name, and are
            # added alongside the rest.
            secret_patterns.update(custom["secret_patterns"])
            pii_patterns.update(custom["pii_patterns"])
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
