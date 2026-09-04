"""
Unit tests for the canary, the detector cache, the detector evaluator, the
audit schema version, and PipelineConfig.
"""

from __future__ import annotations

import json

import pytest

from llm_security_pipeline import (
    AUDIT_SCHEMA_VERSION,
    ENFORCING,
    CallableDetector,
    Canary,
    DetectorEnsemble,
    DetectorEvaluator,
    FailurePolicy,
    InMemoryMetricsSink,
    OutputGuard,
    PipelineConfig,
    Registration,
    SecurityPipeline,
    ThresholdConfig,
    smoke_corpus,
)
from llm_security_pipeline.pipeline import AuditLogger


class Collector(AuditLogger):
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def log(self, event_type, data):
        self.events.append((event_type, data))


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_a_canary_is_unguessable_and_letters_only():
    """Letters only so a leak does not also trip the phone-number
    detector and muddy the audit record with a second category."""
    a, b = Canary.generate(), Canary.generate()
    assert a.token != b.token
    assert a.token.split("-", 1)[1].isalpha()


def test_the_planted_prompt_contains_the_token_and_tells_the_model_not_to_repeat_it():
    canary = Canary.generate()
    planted = canary.plant("You are a bot.")
    assert planted.startswith("You are a bot.")
    assert canary.token in planted
    assert "never appear" in planted


async def test_a_canary_in_the_output_is_a_certain_leak():
    """The overlap heuristic is a probability; this is not."""
    pipeline = SecurityPipeline(session_identity="untrusted", system_prompt="hi", canary=True)

    result = await pipeline.post_process(f"my instructions mention {pipeline.canary.token}")

    assert result.blocked is True
    assert "system_prompt_canary" in result.scan.secret_findings


async def test_a_canary_is_caught_mid_stream():
    pipeline = SecurityPipeline(session_identity="untrusted", system_prompt="hi", canary=True)

    async def source():
        for i in range(0, 60, 4):
            yield f"leak {pipeline.canary.token} end"[i : i + 4]

    guarded = pipeline.guard_stream(source())
    chunks = [c async for c in guarded]

    assert guarded.blocked is True
    assert pipeline.canary.token[:8] not in "".join(chunks)


async def test_the_pipeline_exposes_the_prompt_that_must_actually_be_sent():
    pipeline = SecurityPipeline(session_identity="untrusted", system_prompt="hi", canary=True)
    assert pipeline.canary.token in pipeline.planted_system_prompt
    assert pipeline.system_prompt == "hi"

    plain = SecurityPipeline(session_identity="untrusted", system_prompt="hi")
    assert plain.planted_system_prompt == "hi"


def test_a_canary_without_a_system_prompt_is_refused():
    with pytest.raises(ValueError, match="system_prompt"):
        SecurityPipeline(session_identity="untrusted", canary=True)


async def test_a_caller_supplied_output_guard_learns_about_the_canary():
    """Planted and unwatched is the worst of both."""
    guard = OutputGuard()
    pipeline = SecurityPipeline(
        session_identity="untrusted", system_prompt="hi", canary=True, output_guard=guard,
    )
    assert pipeline.canary in guard.canaries


# ---------------------------------------------------------------------------
# Detector cache
# ---------------------------------------------------------------------------


async def test_the_cache_is_off_by_default():
    calls = 0

    def clf(_t):
        nonlocal calls
        calls += 1
        return 0.1

    ensemble = DetectorEnsemble([Registration(CallableDetector("c", clf))])
    await ensemble.score("x")
    await ensemble.score("x")
    assert calls == 2


async def test_the_cache_serves_repeats_and_is_bounded():
    calls = 0

    def clf(_t):
        nonlocal calls
        calls += 1
        return 0.1

    ensemble = DetectorEnsemble([Registration(CallableDetector("c", clf))], cache_size=2)
    for text in ["a", "a", "b", "c", "a"]:  # "a" evicted by the time it recurs
        await ensemble.score(text)
    assert calls == 4
    assert ensemble.cache_hits == 1


async def test_errors_are_never_cached():
    """A failed call should be retried, not remembered."""
    attempts = 0

    def flaky(_t):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first time only")
        return 0.2

    ensemble = DetectorEnsemble([Registration(CallableDetector("c", flaky))], cache_size=10)
    first = await ensemble.score("x")
    second = await ensemble.score("x")
    assert "c" in first.errors
    assert second.results and second.results[0].score == 0.2


async def test_the_cache_key_includes_the_registration():
    """Same text, different mode or weight, different answer."""
    ensemble = DetectorEnsemble([Registration(CallableDetector("c", lambda _t: 0.8))], cache_size=10)
    await ensemble.score("x")
    a = ensemble._cache_key("x")
    ensemble.registrations[0] = Registration(CallableDetector("c", lambda _t: 0.8), mode=ENFORCING)
    assert ensemble._cache_key("x") != a


# ---------------------------------------------------------------------------
# One broken detector must not switch off the others
# ---------------------------------------------------------------------------


async def test_a_broken_detector_does_not_trip_the_breaker_for_a_healthy_one():
    """Found by an evaluator smoke test: with one breaker per category, a
    detector that always fails opened the circuit for every detector."""
    healthy_calls = 0

    def healthy(_t):
        nonlocal healthy_calls
        healthy_calls += 1
        return 0.3

    def broken(_t):
        raise RuntimeError("down")

    ensemble = DetectorEnsemble(
        [Registration(CallableDetector("ok", healthy)), Registration(CallableDetector("bad", broken))],
        failure_policy=FailurePolicy(failure_threshold=2, recovery_seconds=60),
    )
    for _ in range(10):
        await ensemble.score("x")

    assert healthy_calls == 10


# ---------------------------------------------------------------------------
# Detector evaluator
# ---------------------------------------------------------------------------


async def test_the_evaluator_scores_every_signal_separately():
    corpus = smoke_corpus()
    attacks = {s.text for s in corpus if s.label == "attack"}
    ensemble = DetectorEnsemble([
        Registration(CallableDetector("oracle", lambda t: 0.9 if t in attacks else 0.1)),
    ])

    scores = await DetectorEvaluator(ensemble).score(corpus)

    assert set(scores.signals) == {"lexical", "combined", "oracle"}
    assert len(scores.by_signal["oracle"]) == len(corpus)


async def test_a_perfect_detector_is_recommended_and_a_dead_one_is_not():
    corpus = smoke_corpus()
    attacks = {s.text for s in corpus if s.label == "attack"}

    def broken(_t):
        raise RuntimeError("x")

    ensemble = DetectorEnsemble([
        Registration(CallableDetector("oracle", lambda t: 0.9 if t in attacks else 0.1)),
        Registration(CallableDetector("dead", broken)),
    ])
    evaluator = DetectorEvaluator(ensemble)
    scores = await evaluator.score(corpus)

    oracle = evaluator.recommend(scores, "oracle")
    assert oracle.recall == 1.0 and oracle.false_positive_rate == 0.0

    report = evaluator.promotion_report(scores)
    assert "oracle" in report and "recall over lexical" in report
    assert "dead" in report and "no data" in report


async def test_a_partially_errored_detector_says_how_much_of_the_corpus_it_saw():
    """A recall over a subset is not comparable to one over the whole."""
    corpus = smoke_corpus()
    seen = 0

    def flaky(_t):
        nonlocal seen
        seen += 1
        if seen % 3 == 0:
            raise RuntimeError("x")
        return 0.5

    ensemble = DetectorEnsemble(
        [Registration(CallableDetector("flaky", flaky))],
        failure_policy=FailurePolicy(failure_threshold=1000),
    )
    evaluator = DetectorEvaluator(ensemble)
    report = evaluator.promotion_report(await evaluator.score(corpus))

    assert "only" in report and "/24 scored" in report


def test_the_lexical_evaluator_still_works_after_the_refactor():
    from llm_security_pipeline import Evaluator

    report = Evaluator().recommend(smoke_corpus())
    assert 0.0 <= report.threshold <= 1.0


# ---------------------------------------------------------------------------
# Audit schema version
# ---------------------------------------------------------------------------


async def test_every_audit_event_carries_a_schema_version():
    audit = Collector()
    pipeline = SecurityPipeline(session_identity="untrusted", audit_logger=audit)
    await pipeline.pre_process("hello", session_id="s1")
    await pipeline.post_process("world")
    await pipeline.pre_process_external("doc", source_id="rag:1")

    assert audit.events
    assert all(data["schema_version"] == AUDIT_SCHEMA_VERSION for _, data in audit.events)


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


def test_config_round_trips_through_json():
    config = PipelineConfig(
        session_identity="authenticated",
        enforcement="shadow",
        system_prompt="hi",
        canary=True,
        thresholds=ThresholdConfig(input_risk=0.4),
        failure_policy=FailurePolicy(rate_limit="closed"),
    )
    assert PipelineConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_a_misspelled_setting_is_an_error_not_a_default():
    """A typo that silently falls back to the default is a security
    posture that is not the one written down."""
    with pytest.raises(ValueError, match="unknown key"):
        PipelineConfig.from_dict({"session_identity": "untrusted", "tresholds": {}})


@pytest.mark.parametrize(
    "data",
    [
        {"session_identity": "root"},
        {"session_identity": "untrusted", "enforcement": "sometimes"},
        {"session_identity": "untrusted", "canary": True},
        {"session_identity": "untrusted", "thresholds": {"input_risk": 1.5}},
    ],
)
def test_invalid_config_is_refused_at_construction(data):
    with pytest.raises(ValueError):
        PipelineConfig.from_dict(data)


def test_session_identity_has_no_default_in_config_either():
    with pytest.raises(TypeError):
        PipelineConfig()  # type: ignore[call-arg]


async def test_from_config_builds_the_same_pipeline_as_the_kwargs():
    config = PipelineConfig(session_identity="untrusted", enforcement="shadow",
                            thresholds=ThresholdConfig(input_risk=0.3))
    a = SecurityPipeline.from_config(config, metrics=InMemoryMetricsSink())
    b = SecurityPipeline(session_identity="untrusted", enforcement="shadow",
                         input_risk_threshold=0.3, metrics=InMemoryMetricsSink())
    assert a.config_summary == b.config_summary


def test_a_setting_passed_beside_a_config_is_refused():
    config = PipelineConfig(session_identity="untrusted")
    with pytest.raises(ValueError, match="belong in the PipelineConfig"):
        SecurityPipeline.from_config(config, enforcement="enforce")


def test_config_summary_is_the_posture_not_the_objects():
    pipeline = SecurityPipeline(session_identity="untrusted", system_prompt="hi", canary=True)
    summary = pipeline.config_summary
    assert summary["canary"] is True
    assert pipeline.canary.token not in json.dumps(summary)
    assert "active_key_id" in summary
