"""
Unit tests for PatternRegistry / load_pattern_file: no infrastructure
needed, just JSON files on disk (via pytest's tmp_path).
"""

from __future__ import annotations

import json

import pytest

from llm_security_pipeline import PatternConfigError, PatternRegistry


def _write_json(path, data: dict) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_load_defaults_covers_all_three_categories():
    registry = PatternRegistry.load()

    assert "anthropic_api_key" in registry.secret_patterns
    assert "email" in registry.pii_patterns
    assert "en" in registry.injection_patterns
    assert "en_ignore_previous_instructions" in registry.injection_patterns["en"]
    assert "it" in registry.injection_patterns
    assert "it_ignora_istruzioni_precedenti" in registry.injection_patterns["it"]


def test_custom_config_adds_new_name_alongside_defaults(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [{"name": "my_custom_token", "pattern": "custom-[0-9]{6}", "flags": []}],
        "pii_patterns": [],
        "injection_patterns": [
            {"name": "en_my_custom_phrase", "lang": "en", "pattern": "do the forbidden thing", "flags": ["IGNORECASE"]},
        ],
    })

    registry = PatternRegistry.load(custom_config_path=custom_path)

    # New names sit alongside the defaults rather than replacing them.
    assert "my_custom_token" in registry.secret_patterns
    assert "anthropic_api_key" in registry.secret_patterns
    assert "en_my_custom_phrase" in registry.injection_patterns["en"]
    assert "en_ignore_previous_instructions" in registry.injection_patterns["en"]


def test_custom_config_overrides_default_with_same_name(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [{"name": "anthropic_api_key", "pattern": "OVERRIDDEN-[0-9]+", "flags": []}],
        "pii_patterns": [],
        "injection_patterns": [],
    })

    registry = PatternRegistry.load(custom_config_path=custom_path)

    assert registry.secret_patterns["anthropic_api_key"].pattern == "OVERRIDDEN-[0-9]+"


def test_custom_only_config_ignores_bundled_defaults(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [{"name": "only_mine", "pattern": "x", "flags": []}],
        "pii_patterns": [],
        "injection_patterns": [],
    })

    registry = PatternRegistry.load(custom_config_path=custom_path, include_defaults=False)

    assert list(registry.secret_patterns) == ["only_mine"]
    assert registry.pii_patterns == {}
    assert registry.injection_patterns == {}


def test_pattern_entry_rejects_unknown_field(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [{"name": "x", "pattern": "x", "flags": [], "unexpected_field": 1}],
        "pii_patterns": [],
        "injection_patterns": [],
    })

    with pytest.raises(PatternConfigError, match="unexpected_field"):
        PatternRegistry.load(custom_config_path=custom_path)


def test_pattern_entry_rejects_empty_name(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [{"name": "", "pattern": "x", "flags": []}],
        "pii_patterns": [],
        "injection_patterns": [],
    })

    with pytest.raises(PatternConfigError, match=r"secret_patterns\.0\.name"):
        PatternRegistry.load(custom_config_path=custom_path)


def test_malformed_json_raises_pattern_config_error(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(PatternConfigError, match="Malformed JSON"):
        PatternRegistry.load(custom_config_path=str(bad_path))


def test_missing_file_raises_pattern_config_error(tmp_path):
    with pytest.raises(PatternConfigError, match="not found"):
        PatternRegistry.load(custom_config_path=str(tmp_path / "missing.json"))


def test_pattern_entry_missing_name_raises(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [{"pattern": "x", "flags": []}],
        "pii_patterns": [],
        "injection_patterns": [],
    })

    with pytest.raises(PatternConfigError, match=r"secret_patterns\.0\.name"):
        PatternRegistry.load(custom_config_path=custom_path)


def test_injection_entry_missing_lang_raises(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [],
        "pii_patterns": [],
        "injection_patterns": [{"name": "no_lang", "pattern": "x"}],
    })

    with pytest.raises(PatternConfigError, match=r"injection_patterns\.0\.lang"):
        PatternRegistry.load(custom_config_path=custom_path)


def test_unknown_flag_raises(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [{"name": "x", "pattern": "x", "flags": ["NOT_A_REAL_FLAG"]}],
        "pii_patterns": [],
        "injection_patterns": [],
    })

    with pytest.raises(PatternConfigError, match="Unknown regex flag"):
        PatternRegistry.load(custom_config_path=custom_path)


def test_invalid_regex_raises(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [{"name": "x", "pattern": "(unclosed", "flags": []}],
        "pii_patterns": [],
        "injection_patterns": [],
    })

    with pytest.raises(PatternConfigError, match="Invalid regular expression"):
        PatternRegistry.load(custom_config_path=custom_path)


def test_zero_patterns_raises(tmp_path):
    custom_path = _write_json(tmp_path / "custom.json", {
        "secret_patterns": [], "pii_patterns": [], "injection_patterns": [],
    })

    with pytest.raises(PatternConfigError, match="zero patterns"):
        PatternRegistry.load(custom_config_path=custom_path, include_defaults=False)