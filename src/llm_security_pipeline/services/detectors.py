"""
detectors.py
A seam for detection that is not lexical.

Everything else in this library decides by shape: a pattern matched, a
codepoint was invisible, a base64 blob decoded into an instruction. The
README is honest that this is bypassed by paraphrasing — "disregard your
earlier guidance" is not in any phrase list, and putting it there only
moves the problem to the next wording.

Closing that gap needs a model, and this library does not ship one. That is
the same position `MediaScanner` takes about OCR: extraction is yours,
scoring is ours. Bundling a classifier would mean pinning a runtime, a set
of weights and a licence into a dependency-light security library, and
freezing a fast-moving choice on everyone who installs it. So what is
provided is the socket: a protocol, concurrent execution, failure handling
that does not lie, and a defined way for the score to reach the decision.

--- Why a new detector is advisory until you say otherwise ----------------

A classifier's calibration is a property of your traffic, not of the
classifier. A model with an excellent published F1 will still have a
threshold that is wrong for your users on the day you install it, and the
failure mode of getting that wrong is blocking real customers.

So a registered detector defaults to `ADVISORY`: it runs, it is scored, it
is written to the audit record, and it does not change any decision.
Measure it on your own traffic with `llm_security_pipeline.evaluation`,
pick a threshold, then switch it to `ENFORCING`. This is the same shape as
the pipeline's shadow mode, one detector at a time.

--- Why scores combine by max and not by sum ------------------------------

A detector firing on the same sentence the lexical scan already matched is
the same evidence read twice. Adding them would let two weak, correlated
signals reach a threshold neither deserved, and the effect grows with every
detector added — an ensemble would drift towards blocking everything.
`max` keeps a confident detector able to raise the verdict alone while
refusing to manufacture confidence out of agreement. Pass `combine=` if
your calibration says otherwise.

--- Why a failed detector contributes nothing, not zero -------------------

Same rule as the media scanner: 0.0 means "looked, found nothing", and a
detector that timed out has not looked. It is recorded in `errors`,
reported as a degradation, and excluded from the combination, so a model
server going down cannot quietly lower every risk score in the system.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..resilience import DETECTOR, Degraded, FailurePolicy, ResilientBackend

ADVISORY = "advisory"
ENFORCING = "enforcing"
_MODES = (ADVISORY, ENFORCING)


@dataclass(frozen=True)
class DetectorResult:
    """One detector's opinion about one piece of text.

    `score` is on the same 0..1 scale as the lexical risk score, which is
    what makes them combinable at all. A detector whose native output is a
    logit or a distance is expected to map it before returning; the mapping
    is a calibration decision and belongs with whoever owns the model.
    """

    detector: str
    score: float
    label: str | None = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Detector {self.detector!r} returned {self.score}, outside 0..1. "
                "Map your model's native output onto that scale before returning "
                "it; the pipeline compares it against thresholds on that scale."
            )


@runtime_checkable
class SemanticDetector(Protocol):
    """Anything that can score text for hostile intent.

    Implementations may be sync or async and the ensemble handles both. A
    sync one is run on a worker thread rather than inline, because an
    inference call blocking the event loop stalls every other request in
    the process; local inference runtimes release the GIL, so a thread is
    the right place for them. Network-backed judges should be async and do
    their own I/O.
    """

    name: str

    def score(self, text: str) -> DetectorResult | float:
        ...


@dataclass
class CallableDetector:
    """Adapter so a plain function is a detector.

        ensemble = DetectorEnsemble([
            Registration(CallableDetector("my-clf", classify), mode=ENFORCING),
        ])

    `classify` may be sync or async and may return a float or a
    DetectorResult.
    """

    name: str
    fn: Callable[[str], object]

    def score(self, text: str) -> object:
        return self.fn(text)


@dataclass
class Registration:
    """A detector plus how much this deployment trusts it.

    `weight` scales the score before it is combined, so a detector known to
    be over-eager can be kept without letting it decide alone. `threshold`
    silences it below a level: many classifiers emit a noisy floor on
    perfectly ordinary text, and reporting that as risk 0.12 on every turn
    trains people to ignore the field.
    """

    detector: SemanticDetector
    mode: str = ADVISORY
    weight: float = 1.0
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"Registration.mode must be one of {_MODES}, got {self.mode!r}.")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Registration.weight must be between 0 and 1.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Registration.threshold must be between 0 and 1.")
        if not getattr(self.detector, "name", None):
            raise ValueError("A detector must have a non-empty `name`; it labels the audit record.")

    @property
    def name(self) -> str:
        return self.detector.name

    def contribution(self, result: DetectorResult) -> float:
        """The weighted score, or 0.0 if it did not clear its threshold."""
        if result.score < self.threshold:
            return 0.0
        return result.score * self.weight


@dataclass
class EnsembleResult:
    """What every detector said, and what it adds up to.

    `enforcing_score` is the only number that can change a decision.
    `advisory_score` is what the advisory detectors would have contributed
    had they been enforcing, which is the number to look at when deciding
    whether to promote one.
    """

    results: list[DetectorResult] = field(default_factory=list)
    # Detector name -> what went wrong. A detector in here contributed
    # nothing and was NOT counted as 0.0.
    errors: dict[str, str] = field(default_factory=dict)
    enforcing_score: float = 0.0
    advisory_score: float = 0.0

    @property
    def ran(self) -> bool:
        return bool(self.results) or bool(self.errors)

    @property
    def labels(self) -> list[str]:
        return [r.label for r in self.results if r.label]

    def as_audit(self) -> dict:
        return {
            "scores": {r.detector: round(r.score, 3) for r in self.results},
            "labels": {r.detector: r.label for r in self.results if r.label},
            "enforcing_score": round(self.enforcing_score, 3),
            "advisory_score": round(self.advisory_score, 3),
            "errors": dict(self.errors),
        }


def combine_max(lexical_score: float, ensemble: EnsembleResult) -> float:
    """Default combiner. See the module docstring for why it is not a sum."""
    return max(lexical_score, ensemble.enforcing_score)


class DetectorEnsemble:
    """Runs registered detectors concurrently and merges their verdicts.

    Concurrently with each other, and — at the pipeline level — concurrently
    with the lexical scan, because a network-backed judge adds its full
    latency to every turn if it is awaited in sequence.
    """

    def __init__(
        self,
        registrations: list[Registration] | None = None,
        resilience: ResilientBackend | None = None,
        failure_policy: FailurePolicy | None = None,
        combine: Callable[[float, EnsembleResult], float] = combine_max,
        cache_size: int = 0,
        cache_ttl_seconds: float = 300.0,
    ):
        # Off by default. Worth turning on in front of a network-backed
        # judge: the same chunk retrieved by two queries, or the same
        # message on a retry, is otherwise paid for twice. Keyed by a hash
        # of the text so the cache holds no user content, bounded by
        # `cache_size` (LRU) and `cache_ttl_seconds` so a detector that is
        # re-tuned is not answered from before the re-tune forever.
        # Errors are never cached: a failed call should be retried, not
        # remembered.
        self._cache: OrderedDict[str, tuple[float, EnsembleResult]] = OrderedDict()
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl_seconds
        self.cache_hits = 0
        self.registrations = list(registrations or [])
        names = [r.name for r in self.registrations]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Detector names must be unique; they key the audit record. "
                f"Duplicated: {', '.join(sorted(duplicates))}."
            )
        self._resilience = resilience or ResilientBackend(failure_policy)
        self.combine = combine

    def __bool__(self) -> bool:
        return bool(self.registrations)

    @property
    def enforcing(self) -> list[str]:
        return [r.name for r in self.registrations if r.mode == ENFORCING]

    async def score(self, text: str) -> EnsembleResult:
        if not self.registrations:
            return EnsembleResult()

        key = self._cache_key(text) if self._cache_size else None
        if key is not None:
            cached = self._cache.get(key)
            if cached is not None and time.monotonic() - cached[0] < self._cache_ttl:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                return cached[1]

        result = await self._score_uncached(text)

        if key is not None and not result.errors:
            self._cache[key] = (time.monotonic(), result)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def _cache_key(self, text: str) -> str:
        # Registrations are part of the key: the same text scored by a
        # different set of detectors, or the same detector in a different
        # mode, is a different answer.
        config = "|".join(f"{r.name}:{r.mode}:{r.weight}:{r.threshold}" for r in self.registrations)
        return hashlib.sha256(f"{config}\x00{text}".encode()).hexdigest()

    async def _score_uncached(self, text: str) -> EnsembleResult:
        outcomes = await asyncio.gather(
            *(self._run_one(r, text) for r in self.registrations)
        )

        result = EnsembleResult()
        for registration, outcome in zip(self.registrations, outcomes, strict=True):
            if isinstance(outcome, str):
                # A reason, not a score. Excluded from the arithmetic on
                # purpose: a detector that did not answer has no opinion,
                # and 0.0 is an opinion.
                result.errors[registration.name] = outcome
                continue
            result.results.append(outcome)
            contribution = registration.contribution(outcome)
            if registration.mode == ENFORCING:
                result.enforcing_score = max(result.enforcing_score, contribution)
            else:
                result.advisory_score = max(result.advisory_score, contribution)
        return result

    async def _run_one(self, registration: Registration, text: str) -> "DetectorResult | str":
        """Run one detector, returning its result or a reason it has none.

        Never returns a score for a detector that failed: the caller puts
        the reason in `errors` and leaves it out of the arithmetic.
        """
        captured: dict[str, str] = {}

        async def call() -> object:
            try:
                if _is_async(registration.detector):
                    outcome = registration.detector.score(text)
                else:
                    # Almost certainly local inference. Inline, it would
                    # block the event loop for every other request in this
                    # process; inference runtimes release the GIL, so a
                    # thread is the right place for it.
                    outcome = await asyncio.to_thread(registration.detector.score, text)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                return outcome
            except Exception as exc:
                # Kept so the audit record says what broke rather than the
                # generic "unavailable" the policy layer would report.
                captured["error"] = f"{type(exc).__name__}: {exc}"[:200]
                raise

        raw = await self._resilience.run(DETECTOR, call, instance=registration.name)
        if isinstance(raw, Degraded):
            return captured.get("error", "unavailable (timed out or circuit open)")
        return _normalize(registration.name, raw)


def _is_async(detector: SemanticDetector) -> bool:
    fn = getattr(detector, "score", None)
    if isinstance(detector, CallableDetector):
        fn = detector.fn
    return inspect.iscoroutinefunction(fn)


def _normalize(name: str, raw: object) -> DetectorResult | str:
    """Accept a float or a DetectorResult; reject anything else clearly.

    Returned as a string on failure rather than raised, so one detector
    with a broken contract does not take down the scan of a request the
    other detectors handled fine.
    """
    if isinstance(raw, DetectorResult):
        return raw
    if isinstance(raw, bool):
        return "returned a bool; expected a float in 0..1 or a DetectorResult"
    if isinstance(raw, (int, float)):
        try:
            return DetectorResult(detector=name, score=float(raw))
        except ValueError as exc:
            return str(exc)
    return f"returned {type(raw).__name__}; expected a float in 0..1 or a DetectorResult"
