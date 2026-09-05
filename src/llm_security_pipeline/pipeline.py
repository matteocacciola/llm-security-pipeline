"""
pipeline.py
Top-level async orchestrator for production, multi-process/multi-pod
deployments. Chains: input sanitization (direct and indirect prompt
injection), tool-call scope gating, session-level rate limiting, and output
checks, with structured audit logging.

--- Why async, and where the parallelism actually is -----------------------

There are two different kinds of work in this pipeline, and they need two
different parallelization strategies:

1. I/O-bound work (talking to Redis): naturally concurrent under asyncio.
   Checking a session's request budget and recording its risk score are
   independent Redis round-trips and are dispatched together with
   `asyncio.gather` wherever they don't depend on each other.

2. CPU-bound work (the regex-heavy sanitizer/output scans): Python's GIL
   means asyncio alone does NOT parallelize this — coroutines still take
   turns on a single core. Real parallelism for CPU-bound work requires a
   ProcessPoolExecutor (separate OS processes, separate GIL each).

   This pipeline only offloads scanning to the process pool where it
   actually pays off: scanning a BATCH of several external content chunks
   (`pre_process_external_batch`), where N independent scans can run on N
   cores at once. For a single, typically short, direct user message
   (`pre_process`), the serialization/IPC overhead of a process pool round
   trip is usually *slower* than just running the regex scan inline — so
   that path stays synchronous by default, with an opt-in size threshold
   for the rare case of a very large pasted document. Blindly parallelizing
   everything "because parallelism" would make the common case slower, not
   faster; the point is to parallelize where it helps and stay direct where
   it doesn't.

   Note that true multi-core throughput for the *many small requests*
   case (the normal chat workload) comes from running multiple OS
   processes of your application itself (e.g. several uvicorn/gunicorn
   workers, or several pods) — that's why the Redis-backed state in this
   file exists at all: so those independent processes share one consistent
   view of rate limits and token usage instead of each enforcing its own.

--- Typical usage ------------------------------------------------------

    from llm_security_pipeline import SecurityPipeline, RedisStateBackend

    # Redis is one implementation of the required property (centralized,
    # atomic shared state) — not the only one. Any StateBackend works:
    # implement NonceStore/SessionStore on Postgres, DynamoDB, etc. and
    # pass StateBackend(nonce_store=..., session_store=...) instead.
    backend = RedisStateBackend.from_url("redis://redis-service:6379/0")

    async with SecurityPipeline(
        system_prompt=SYSTEM_PROMPT,
        state_backend=backend,           # wires cross-process scope guard + rate limiter
    ) as pipeline:

        session_id = "user-123-session-456"

        pre = await pipeline.pre_process(user_input, session_id=session_id)
        if pre.blocked:
            return "Request blocked for security reasons."

        chunks = [(chunk_text, f"rag:{i}") for i, chunk_text in enumerate(retrieved_chunks)]
        external_results = await pipeline.pre_process_external_batch(chunks)

        prompt = build_prompt(pre.sanitized, external_results)
        model_output = await call_llm(prompt)

        post = await pipeline.post_process(model_output)
        return post.safe_text
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import inspect
import logging
import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from collections.abc import AsyncIterator, Awaitable
from typing import TYPE_CHECKING, Any, Callable

from .services import (
    ExfilGuard,
    IngestGuard,
    IngestVerdict,
    MediaScanner,
    MediaScanResult,
    RetrievalVerdict,
    ExfilScanResult,
    ExfilAttemptBlocked,
    Sanitizer,
    SanitizationResult,
    ScopeGuard,
    SigningKeyring,
    OutputGuard,
    OutputScanResult,
    SessionRateLimiter,
    SessionLimits,
    RateLimitExceeded,
)
from .services.output_guard import Canary, clean_forbidden_literals
from .services.scope_guard import DEFAULT_REVOCATION_TTL
from .services.detectors import (
    DetectorEnsemble,
    EnsembleResult,
    Registration,
)
from .services.streaming_guard import (
    DEFAULT_HOLDBACK_CHARS,
    DEFAULT_MIN_CHUNK_CHARS,
    StreamDelta,
    StreamingOutputGuard,
)
from .metrics import MetricsSink, SafeMetricsSink
from .tracing import SafeTracer, Tracer
from .resilience import (
    AUDIT,
    OPERATIONS,
    Degradation,
    FailurePolicy,
    ResilientBackend,
)
from .sessions.stores import ProvenanceRecord
from .state_backend import StateBackend, RedisStateBackend

if TYPE_CHECKING:
    from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Stamped on every audit event. Bump it when a field is added, renamed or
# changes meaning, so a consumer parsing the stream can branch on it instead
# of discovering the change when its parser breaks. History in CHANGELOG.md.
AUDIT_SCHEMA_VERSION = 4

# Degradations collected for the request currently being handled. A
# ContextVar rather than an attribute because one SecurityPipeline serves
# many concurrent requests, and a shared list would hand one caller another
# caller's failures.
_current_degradations: contextvars.ContextVar[list[Degradation] | None] = (
    contextvars.ContextVar("llm_security_pipeline_degradations", default=None)
)

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover
    Redis = None  # type: ignore


@dataclass
class PreProcessResult:
    sanitized: SanitizationResult
    blocked: bool
    would_block: bool = False
    rate_limited: bool = False
    # What the semantic detectors said, if any are registered. None when
    # none are. `sanitized.risk_score` stays the LEXICAL score whatever the
    # detectors report — conflating them would make the audit record
    # unable to answer "which signal actually fired".
    detectors: EnsembleResult | None = None
    # The score the block decision was taken on: the lexical score merged
    # with the enforcing detectors. Equal to the lexical score when no
    # detector is enforcing.
    combined_risk_score: float = 0.0
    # Guards that could not run because their backend was unreachable, and
    # were allowed through by the failure policy. Empty on the happy path.
    # A caller that ignores this is running unguarded without knowing it,
    # which is the failure mode fail-open exists to make survivable rather
    # than invisible.
    degraded: tuple[Degradation, ...] = ()


@dataclass
class PostProcessResult:
    scan: OutputScanResult
    safe_text: str
    blocked: bool
    exfil: ExfilScanResult | None = None
    # In shadow mode `blocked` is always False while `would_block` records
    # what enforcement would have done. Equal to `blocked` otherwise.
    would_block: bool = False
    degraded: tuple[Degradation, ...] = ()


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
# Cloud-native default: structured JSON lines to stdout. Containerized
# deployments are expected to ship stdout to a log aggregator (CloudWatch,
# Stackdriver, Loki, etc.) rather than relying on a local file, which
# doesn't survive pod restarts and isn't centralized across instances.
# A Redis Streams logger is provided for cases where you want a queryable,
# centralized audit trail without standing up a separate log pipeline.

class AuditLogger:
    async def log(self, event_type: str, data: dict) -> None:
        raise NotImplementedError


class StdoutAuditLogger(AuditLogger):
    async def log(self, event_type: str, data: dict) -> None:
        record = {"ts": time.time(), "event": event_type, "data": data}
        sys.stdout.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()


class RedisStreamAuditLogger(AuditLogger):
    """Writes audit events to a Redis Stream, so every process/pod appends
    to one centralized, ordered, consumable log via XADD — useful when you
    want to tail or process security events (e.g. feed a SIEM) without a
    separate log pipeline."""

    def __init__(self, redis_client: "Redis", stream_name: str = "sentinel:audit", maxlen: int = 100_000):
        self._redis = redis_client
        self._stream_name = stream_name
        self._maxlen = maxlen

    async def log(self, event_type: str, data: dict) -> None:
        payload = {"event": event_type, "data": json.dumps(data, ensure_ascii=False, default=str)}
        # The redis-py stubs type the mapping more narrowly than the server
        # accepts; both values here are str, which is a valid field value.
        await self._redis.xadd(
            self._stream_name,
            payload,  # type: ignore[arg-type]
            maxlen=self._maxlen,
            approximate=True,
        )


# ---------------------------------------------------------------------------
# Process-pool worker
# ---------------------------------------------------------------------------
# Module-level so it is picklable, and combined so that offloading a large
# output costs one round trip rather than one per guard.

def _scan_output(
    output_guard: OutputGuard,
    exfil_guard: ExfilGuard | None,
    text: str,
    system_prompt: str | None,
    overlap_threshold: float,
    forbidden_literals: tuple[str, ...] = (),
) -> tuple[OutputScanResult, ExfilScanResult | None]:
    result = output_guard.scan(
        text, system_prompt=system_prompt, overlap_threshold=overlap_threshold,
        forbidden_literals=forbidden_literals,
    )
    # The side-channel scan runs over the REDACTED text: a secret the output
    # guard has already replaced can no longer be smuggled out in a URL, and
    # scanning the original would re-flag it.
    exfil = exfil_guard.scan(result.redacted_text) if exfil_guard is not None else None
    return result, exfil


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ToolResultBlocked(Exception):
    """A tool ran, and what it returned looked like an injection.

    Not a ScopeError: the call was authorized and happened. What is
    refused is feeding its output to the model, which is the whole point —
    a tool that fetches a web page returns the web page, and the web page
    is the indirect-injection surface this library exists for.
    `scan` carries the verdict; `output` the raw result, for a caller that
    wants to log it or return it to the user without passing it to the
    model.
    """

    def __init__(self, action: str, scan: SanitizationResult, output: object):
        self.action = action
        self.scan = scan
        self.output = output
        super().__init__(
            f"Result of tool {action!r} was blocked before reaching the model "
            f"(risk {scan.risk_score:.2f}; {', '.join(scan.matched_patterns) or 'detector'})."
        )


def _strings_in(value: object, _depth: int = 0) -> list[str]:
    """Every string a tool result carries, however nested. What a model
    reads is text; a dict of text is a dict of injection surfaces."""
    if _depth > 8:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        try:
            return [value.decode("utf-8")]
        except UnicodeDecodeError:
            return []
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            out.extend(_strings_in(k, _depth + 1))
            out.extend(_strings_in(v, _depth + 1))
        return out
    if isinstance(value, (list, tuple, set)):
        return [t for item in value for t in _strings_in(item, _depth + 1)]
    return []


class _Exhausted:
    """A source with nothing left, so the next pull goes straight to finish()."""

    def __aiter__(self) -> "_Exhausted":
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration


class GuardedStream:
    """A model token stream with the output guard in front of it.

    Async-iterable rather than an async generator function so the verdict
    survives the loop: a generator that has finished has nowhere to put
    "and it was blocked", and a caller who has been forwarding chunks to a
    user needs exactly that after the loop ends.
    """

    def __init__(
        self,
        pipeline: "SecurityPipeline",
        source: AsyncIterator[str],
        holdback_chars: int = DEFAULT_HOLDBACK_CHARS,
        principal: str | None = None,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ):
        self._pipeline = pipeline
        self._source = source
        self._principal = principal
        # Every feed rescans the window, so a model that streams one token
        # at a time would cost window-size times a buffered scan. Chunks
        # are coalesced up to this many characters before feeding; the
        # hold-back already delays emission by more than that, so the
        # user sees nothing different. Off in shadow mode, where timing
        # must be exactly the model's.
        self._min_chunk = 0 if pipeline.enforcement == "shadow" else max(0, min_chunk_chars)
        self._pending = ""
        self._forbidden: tuple[str, ...] | None = None
        shadow = pipeline.enforcement == "shadow"
        self._guard = StreamingOutputGuard(
            output_guard=pipeline.output_guard,
            exfil_guard=pipeline.exfil_guard if pipeline.scan_output_for_exfil else None,
            system_prompt=pipeline.system_prompt,
            overlap_threshold=pipeline.output_overlap_threshold,
            # Shadow mode must not change what the user sees, and that
            # includes when they see it: a hold-back delays every chunk, so
            # observing a stream would silently make it feel slower than
            # the stream it is meant to be measuring. Detection is
            # unaffected — the detection tail is sized independently.
            holdback_chars=0 if shadow else holdback_chars,
            redact_pii=not shadow,
        )
        self._holdback = 0 if shadow else holdback_chars
        self._shadow = shadow
        self.blocked = False
        self.would_block = False
        self.reason: str | None = None
        self.replacement_text: str | None = None
        self.leaked_before_holdback = False
        self._audited = False

    def __aiter__(self) -> "GuardedStream":
        return self

    async def __anext__(self) -> str:
        if self._forbidden is None:
            # Resolved once, on the first pull, because the resolver may be
            # async and __init__ cannot await.
            self._forbidden = await self._pipeline._forbidden_literals(self._principal)
            self._guard.forbidden_literals = self._forbidden
        while True:
            try:
                chunk = await self._source.__anext__()
            except StopAsyncIteration:
                if self._pending:
                    # Whatever was still being coalesced goes in before
                    # finish(), or it would never be scanned or emitted.
                    tail, self._pending = self._pending, ""
                    delta = self._guard.feed(tail)
                    self._record(delta)
                    if self._is_blocking(delta):
                        await self._audit_once()
                        raise
                    if delta.text:
                        # Hand this out now; finish() runs on the next pull.
                        self._source = _Exhausted()
                        return delta.text
                delta = self._guard.finish()
                self._record(delta)
                await self._audit_once()
                if delta.text and not self._is_blocking(delta):
                    return delta.text
                raise
            except BaseException:
                # The model stream failed. Whatever was scanned still gets
                # logged, so a truncated response is not an unlogged one.
                await self._audit_once()
                raise

            if self._min_chunk:
                self._pending += chunk
                if len(self._pending) < self._min_chunk:
                    continue
                chunk, self._pending = self._pending, ""
            delta = self._guard.feed(chunk)
            self._record(delta)
            if self._is_blocking(delta):
                await self._audit_once()
                raise StopAsyncIteration
            # Shadow mode forwards exactly what the model produced.
            text = chunk if self._shadow else delta.text
            if text:
                return text

    def _is_blocking(self, delta: StreamDelta) -> bool:
        return delta.blocked and not self._shadow

    def _record(self, delta: StreamDelta) -> None:
        if not delta.blocked:
            return
        self.would_block = True
        self.reason = delta.reason
        # Meaningless in shadow mode, where the hold-back is off and
        # everything is emitted by design: it would be True on every
        # observation and stop distinguishing anything.
        self.leaked_before_holdback = delta.leaked_before_holdback and not self._shadow
        if not self._shadow:
            self.blocked = True
            self.replacement_text = delta.replacement_text

    async def _audit_once(self) -> None:
        if self._audited:
            return
        self._audited = True
        scan = self._guard.result()
        metrics = self._pipeline.metrics
        metrics.increment(
            "requests_total",
            stage="output_stream",
            outcome="blocked" if self.blocked else "allowed",
            enforcement=self._pipeline.enforcement,
        )
        metrics.observe(
            "risk_score", scan.system_prompt_overlap_score,
            stage="output_stream", kind="system_prompt_overlap",
        )
        for kind, findings in (
            ("secret", scan.secret_findings), ("pii", scan.pii_findings),
        ):
            for category in findings:
                metrics.increment(
                    "findings_total", stage="output_stream", kind=kind, category=category,
                )
        if self.blocked:
            metrics.increment("blocks_total", stage="output_stream", reason=self.reason or "unknown")
        elif self.would_block:
            metrics.increment(
                "would_block_total", stage="output_stream", reason=self.reason or "unknown",
            )
        if self.leaked_before_holdback:
            # Should be zero. A non-zero rate here means holdback_chars is
            # smaller than something the patterns can match, and part of it
            # reached a user before it could be recognized.
            metrics.increment("stream_leaks_total")
        await self._pipeline._audit("output_scan", {
            "streamed": True,
            "secret_categories": list(scan.secret_findings.keys()),
            "pii_categories": list(scan.pii_findings.keys()),
            "system_prompt_overlap_score": scan.system_prompt_overlap_score,
            "blocked": self.blocked,
            "would_block": self.would_block,
            "reason": self.reason,
            # Not the same incident as a clean block: part of the match had
            # already reached the user before it could be recognized.
            "leaked_before_holdback": self.leaked_before_holdback,
            "emitted_chars": self._guard.emitted_chars,
        })

    @property
    def scan(self) -> OutputScanResult:
        return self._guard.result()


class SecurityPipeline:
    def __init__(
        self,
        system_prompt: str | None = None,
        input_risk_threshold: float = 0.6,
        output_overlap_threshold: float = 0.35,
        scope_guard: ScopeGuard | None = None,
        scope_secret_key: bytes | None = None,
        scope_keyring: SigningKeyring | None = None,
        scope_audience: str | None = None,
        scope_keyring_provider: "Callable[[], SigningKeyring] | None" = None,
        scope_key_reload_seconds: float = 60.0,
        sanitizer: Sanitizer | None = None,
        output_guard: OutputGuard | None = None,
        exfil_guard: ExfilGuard | None = None,
        exfil_allowed_hosts: list[str] | None = None,
        scan_output_for_exfil: bool = True,
        scan_tool_call_arguments: bool = True,
        scan_tool_results: bool = True,
        ingest_guard: IngestGuard | None = None,
        media_scanner: MediaScanner | None = None,
        enforcement: str = "enforce",
        session_identity: str | None = None,
        failure_policy: FailurePolicy | None = None,
        detectors: "list[Registration] | DetectorEnsemble | None" = None,
        metrics: MetricsSink | None = None,
        tracer: Tracer | None = None,
        limits_for: "Callable[[str | None], SessionLimits | None] | None" = None,
        foreign_identifiers: "Callable[[str], Any] | None" = None,
        canary: "Canary | bool | None" = None,
        rate_limiter: SessionRateLimiter | None = None,
        audit_logger: AuditLogger | None = None,
        pii_config_path: str | None = None,
        include_default_pii_patterns: bool = True,
        injection_config_path: str | None = None,
        include_default_injection_patterns: bool = True,
        session_limits: SessionLimits | None = None,
        state_backend: StateBackend | None = None,
        process_executor: ProcessPoolExecutor | None = None,
        external_scan_parallel_min_chunks: int = 2,
        large_input_offload_threshold_chars: int = 20_000,
        own_process_executor: bool | None = None,
    ):
        self.system_prompt = system_prompt
        self.input_risk_threshold = input_risk_threshold
        self.output_overlap_threshold = output_overlap_threshold

        # --- Shared-state wiring -------------------------------------------
        # The pipeline's cross-process guarantees require a centralized
        # store with atomic check-and-update semantics — a PROPERTY, not a
        # specific product. `state_backend` is the technology-neutral way
        # to provide it: any StateBackend works (Redis via the ready-made
        # RedisStateBackend, or your own bundle of NonceStore/SessionStore
        # implementations on Postgres, DynamoDB, etc.). With no backend at
        # all, in-memory stores are used: correct for a single process,
        # silently insufficient for multi-process deployments.
        self.state_backend = state_backend

        # --- Identity posture ------------------------------------------
        # `session_id` is a string the caller supplies; nothing in this
        # library can verify it. Every other omission here fails loudly —
        # a missing provenance store raises, a cluster with the wrong key
        # layout raises — but an unauthenticated session id degrades in
        # perfect silence: counters count, logs fill up, and the cumulative
        # risk defence is worth nothing because rotating the id resets it.
        # So the posture has to be stated rather than defaulted into.
        if session_identity not in (None, "untrusted", "authenticated"):
            raise ValueError(
                "session_identity must be 'authenticated', 'untrusted' or None, "
                f"got {session_identity!r}"
            )
        if session_identity is None:
            logger.warning(
                "SecurityPipeline: session_identity was not declared. session_id is "
                "being taken on trust, so a caller that sends a fresh one each turn "
                "resets its cumulative risk and multi-turn detection stops working. "
                "Pass session_identity='authenticated' and a principal= on each call, "
                "or session_identity='untrusted' to accept this knowingly (and pass "
                "actor_id= so risk still accumulates somewhere stable)."
            )
        self.session_identity = session_identity or "untrusted"

        if scope_guard is not None and (
            scope_secret_key is not None or scope_keyring is not None
            or scope_audience is not None or scope_keyring_provider is not None
        ):
            raise ValueError(
                "Pass either scope_guard or scope_secret_key/scope_keyring/scope_audience, "
                "not both — the key and the audience belong to the guard you supplied."
            )
        if scope_secret_key is not None and scope_keyring is not None:
            raise ValueError(
                "Pass either scope_secret_key or scope_keyring, not both. "
                "scope_secret_key is shorthand for a one-key keyring; pass a "
                "SigningKeyring as soon as you need to rotate."
            )
        # A canary is only useful if the prompt sent to the model is the
        # planted one, so the pipeline exposes it: send `planted_system_prompt`.
        if canary is True:
            canary = Canary.generate()
        self.canary: Canary | None = canary or None
        if self.canary is not None and system_prompt is None:
            raise ValueError("A canary needs a system_prompt to be planted in.")

        # Aggregate counters, kept deliberately separate from the audit
        # log: that one is per-request and keeps identifiers, this one is
        # aggregate and carries none. See metrics.py.
        self.metrics = SafeMetricsSink(metrics)
        # Per-tier budgets without N pipelines: called with the principal
        # (None when there is none) and returning a SessionLimits to use
        # for this call, or None for the defaults. Consulted once per
        # request; a callable that raises is a bug and is not swallowed.
        self.limits_for = limits_for
        # Cross-tenant leak detection. Given the principal a response is
        # for, returns the identifiers (emails, account numbers, names)
        # that belong to OTHER principals and must not appear in it. The
        # library cannot know whose data is whose; the application can.
        # Sync or async. What comes back is matched as a secret, so it
        # blocks and redacts through the same path a credential does,
        # buffered or streamed.
        self.foreign_identifiers = foreign_identifiers
        # Spans per guard, under the application's own tracer. Same rule
        # as metrics for what may be an attribute: no identifiers.
        self.tracer = SafeTracer(tracer)

        # One ResilientBackend shared by every guard, so the circuit
        # breakers and the degradation reporting see a single coherent
        # picture instead of each guard forming its own opinion about
        # whether the backend is up.
        self._resilience = ResilientBackend(failure_policy).with_callback(self._on_degraded)
        self.failure_policy = self._resilience.policy

        if scope_guard is None:
            # Without an explicit key the guard generates a private one per
            # process, which means tokens issued here verify nowhere else.
            # ScopeGuard warns about that itself; the warning is escalated
            # here because a state_backend is proof that more than one
            # process is expected to share this state.
            if (scope_secret_key is None and scope_keyring is None
                    and scope_keyring_provider is None and state_backend is not None):
                logger.warning(
                    "SecurityPipeline: a state_backend was configured (so this "
                    "deployment expects several processes to share state) but no "
                    "signing key was given. Capability tokens will be signed "
                    "with a key private to this process and will fail verification "
                    "on every other one. Pass scope_secret_key= (or scope_keyring=) "
                    "from your secret manager, the same value in every process."
                )
            scope_guard = ScopeGuard(
                secret_key=scope_secret_key,
                keyring=scope_keyring,
                audience=scope_audience,
                keyring_provider=scope_keyring_provider,
                reload_interval_seconds=scope_key_reload_seconds,
                resilience=self._resilience,
                nonce_store=state_backend.nonce_store if state_backend is not None else None,
                # An authenticated deployment gets subject-bound tokens by
                # default: a token that names nobody is spendable by anyone
                # who finds it.
                require_subject=self.session_identity == "authenticated",
            )
        self.scope_guard = scope_guard

        if rate_limiter is None:
            rate_limiter = SessionRateLimiter(
                limits=session_limits,
                store=state_backend.session_store if state_backend is not None else None,
                resilience=self._resilience,
            )
        self.rate_limiter = rate_limiter

        self.sanitizer = sanitizer or Sanitizer(
            pattern_config_path=injection_config_path,
            include_default_patterns=include_default_injection_patterns,
        )

        self.output_guard = output_guard or OutputGuard(
            pattern_config_path=pii_config_path,
            include_default_patterns=include_default_pii_patterns,
            canaries=(self.canary,) if self.canary is not None else None,
        )
        if output_guard is not None and self.canary is not None and self.canary not in output_guard.canaries:
            # A caller-supplied guard has to know about the canary too, or
            # the marker is planted and nothing is watching for it.
            output_guard.canaries = (*output_guard.canaries, self.canary)

        # Side-channel guard. Enabled by default with no allowlist, where it
        # still catches the unambiguous cases (data: URIs, embedded
        # credentials, encoded or high-entropy payloads). Passing
        # `exfil_allowed_hosts` is what turns it into a real control: any
        # auto-fetching URL pointing off the list is then a finding on shape
        # alone, which is the version an attacker cannot dress up.
        if exfil_guard is not None and exfil_allowed_hosts is not None:
            raise ValueError("Pass either exfil_guard or exfil_allowed_hosts, not both.")
        self.exfil_guard = exfil_guard or ExfilGuard(allowed_hosts=exfil_allowed_hosts)
        # Two switches rather than one, because the guard has two
        # integration points with different blast radii: rewriting a reply
        # and refusing an action. Either can be turned off without the
        # other, and both default on.
        self.scan_output_for_exfil = scan_output_for_exfil
        self.scan_tool_call_arguments = scan_tool_call_arguments
        self.scan_tool_results = scan_tool_results

        # Shadow mode: everything is scanned, logged and scored, nothing is
        # blocked or rewritten. The point is to calibrate thresholds against
        # real traffic before enforcing them — see evaluation.py — because
        # the alternative is finding out what your false-positive rate is
        # from your users.
        if enforcement not in ("enforce", "shadow"):
            raise ValueError(f"enforcement must be 'enforce' or 'shadow', got {enforcement!r}")
        self.enforcement = enforcement

        # Ingest-time scanning. The guard works without a provenance store;
        # only record/verify need one, and it comes from the state backend
        # when that backend provides it.
        if isinstance(detectors, DetectorEnsemble):
            self.detectors = detectors
            self._adopt_reporting(detectors)
        else:
            self.detectors = DetectorEnsemble(detectors, resilience=self._resilience)
        if self.detectors and not self.detectors.enforcing:
            logger.info(
                "SecurityPipeline: %d semantic detector(s) registered, none enforcing. "
                "They will be scored and audited without changing any decision. "
                "Measure them on your own traffic (llm_security_pipeline.evaluation), "
                "then re-register with mode='enforcing'.",
                len(self.detectors.registrations),
            )

        # A guard supplied by the caller brought its own ResilientBackend,
        # and therefore its own policy — which is theirs to choose and is
        # left alone. What is adopted is the reporting: without this, a
        # degraded check on a caller-supplied rate limiter would reach the
        # log and never the caller, and "fail-open is never silent" would
        # quietly stop being true for anyone who configured their limits the
        # documented way.
        self._adopt_reporting(scope_guard, rate_limiter)

        self.ingest_guard = ingest_guard or IngestGuard(
            sanitizer=self.sanitizer,
            exfil_guard=self.exfil_guard,
            resilience=self._resilience,
            provenance_store=getattr(state_backend, "provenance_store", None),
        )

        # Non-text inputs. The default scanner reads printable strings only;
        # plug in OCR or a vision model via MediaScanner(extractors=[...])
        # to cover text rendered as pixels. See media_guard.py for why no
        # OCR engine is bundled.
        # Same threshold as every other input surface. A media payload was
        # previously judged against the scanner's own default regardless of
        # what the pipeline was configured with.
        self.media_scanner = media_scanner or MediaScanner(
            sanitizer=self.sanitizer, threshold=input_risk_threshold,
        )

        if audit_logger is None:
            # Default to the Redis Streams audit logger only when the
            # backend actually is Redis; any other backend falls back to
            # structured stdout logging (pass audit_logger explicitly to
            # integrate with your own logging/SIEM pipeline).
            if isinstance(state_backend, RedisStateBackend):
                audit_logger = RedisStreamAuditLogger(state_backend.redis_client)
            else:
                audit_logger = StdoutAuditLogger()
        self.audit = audit_logger

        # --- Process pool for genuinely parallel CPU-bound batch work -----
        self._owns_process_executor = own_process_executor if own_process_executor is not None else process_executor is None
        self.process_executor = process_executor or ProcessPoolExecutor(max_workers=os.cpu_count() or 4)
        self.external_scan_parallel_min_chunks = external_scan_parallel_min_chunks
        self.large_input_offload_threshold_chars = large_input_offload_threshold_chars

    @classmethod
    def from_config(cls, config: "PipelineConfig", **objects: Any) -> "SecurityPipeline":
        """Build a pipeline from a PipelineConfig plus the per-process
        objects a config cannot carry (backend, executor, sinks, guards).

        Passing a setting as a keyword here is refused rather than merged:
        with two sources for one value there is no way to tell, reading
        the call, which one won.
        """
        from .config import SETTING_KWARGS

        clash = SETTING_KWARGS & set(objects)
        if clash:
            raise ValueError(
                f"{', '.join(sorted(clash))} are settings and belong in the "
                "PipelineConfig, not alongside it. Only objects (state_backend, "
                "audit_logger, metrics, guards, executors, keys) go here."
            )
        return cls(**config.to_kwargs(), **objects)

    @property
    def config_summary(self) -> dict[str, Any]:
        """The settings this pipeline is running with, as data. For a
        startup log line or a health endpoint: the posture, not the objects."""
        return {
            "session_identity": self.session_identity,
            "enforcement": self.enforcement,
            "input_risk_threshold": self.input_risk_threshold,
            "output_overlap_threshold": self.output_overlap_threshold,
            "canary": self.canary is not None,
            "detectors": {r.name: r.mode for r in self.detectors.registrations},
            "failure_policy": {op: self.failure_policy.decision_for(op) for op in OPERATIONS},
            "signing_key_ids": list(self.scope_guard.keyring.key_ids),
            "audience": self.scope_guard.audience,
            "active_key_id": self.scope_guard.active_key_id,
            "metrics": self.metrics.enabled,
        }

    @property
    def planted_system_prompt(self) -> str | None:
        """The system prompt to actually send to the model.

        With a canary configured this differs from `system_prompt` by one
        planted line, and sending the original instead means the canary
        guards nothing. Without one it is the same string.
        """
        if self.system_prompt is None:
            return None
        if self.canary is None:
            return self.system_prompt
        return self.canary.plant(self.system_prompt)

    def _limits(self, principal: str | None) -> "SessionLimits | None":
        return self.limits_for(principal) if self.limits_for is not None else None

    # -- Tokens: revoke, delegate ------------------------------------------

    async def revoke_subject(self, subject: str, ttl_seconds: int = DEFAULT_REVOCATION_TTL) -> float:
        """Refuse every capability token `subject` holds. See ScopeGuard.revoke_subject."""
        revoked_at = await self.scope_guard.revoke_subject(subject, ttl_seconds)
        await self._audit("subject_revoked", {"subject": subject, "revoked_at": revoked_at})
        self.metrics.increment("requests_total", stage="revocation", outcome="revoked",
                               enforcement=self.enforcement)
        return revoked_at

    def attenuate(self, parent, scopes: list[str], **kwargs):
        """Derive a narrower token for a sub-agent. See ScopeGuard.attenuate."""
        return self.scope_guard.attenuate(parent, scopes, **kwargs)

    # -- Ingest review -----------------------------------------------------

    async def review_queue(self, limit: int = 100):
        return await self.ingest_guard.review_queue(limit)

    async def approve_document(self, document_id: str) -> bool:
        decided = await self.ingest_guard.approve(document_id)
        await self._audit("ingest_review", {"document_id": document_id, "decision": "accept", "applied": decided})
        return decided

    async def reject_document(self, document_id: str) -> bool:
        decided = await self.ingest_guard.reject(document_id)
        await self._audit("ingest_review", {"document_id": document_id, "decision": "reject", "applied": decided})
        return decided

    def _all_resilience(self) -> list[ResilientBackend]:
        seen: list[ResilientBackend] = [self._resilience]
        for guard in (self.rate_limiter, self.scope_guard, self.ingest_guard, self.detectors):
            backend = getattr(guard, "_resilience", None) or getattr(guard, "resilience", None)
            if isinstance(backend, ResilientBackend) and all(backend is not b for b in seen):
                seen.append(backend)
        return seen

    # -- Health ------------------------------------------------------------

    async def health(self, timeout_seconds: float = 2.0) -> dict[str, Any]:
        """What an operator wants from a health endpoint: is each backend
        answering, are any breakers open, and what posture is running.

        Never raises. Each probe is one cheap read against the real store
        (a key that never exists), bounded by `timeout_seconds`, and its
        failure is reported as a string rather than propagated. `status`
        is "ok" when every probe passed and no breaker is open,
        "degraded" otherwise — the same word the failure policy uses,
        because it means the same thing: some checks are not running.
        """
        probes: dict[str, str] = {}

        async def probe(name: str, call) -> None:
            try:
                await asyncio.wait_for(call(), timeout_seconds)
                probes[name] = "ok"
            except Exception as exc:
                probes[name] = f"{type(exc).__name__}: {exc}"[:120]

        await probe("session_store", lambda: self.rate_limiter._store.is_flagged("__health__"))
        await probe("nonce_store", lambda: self.scope_guard._nonce_store.revoked_at("__health__"))
        if self.ingest_guard.provenance_store is not None:
            await probe("provenance_store", lambda: self.ingest_guard.provenance_store.get("__health__"))

        # Every guard's breakers, not just the pipeline's own: a caller-
        # supplied guard keeps its own ResilientBackend (and its own
        # policy), and a breaker open in there is exactly as much "some
        # checks are not running" as one in here.
        open_breakers: dict[str, str] = {}
        for backend in self._all_resilience():
            open_breakers.update(backend.breaker_states())
        healthy = all(v == "ok" for v in probes.values()) and not open_breakers
        return {
            "status": "ok" if healthy else "degraded",
            "backends": probes,
            "open_breakers": open_breakers,
            "posture": self.config_summary,
        }

    # -- Streaming output -------------------------------------------------

    def guard_stream(
        self,
        source: "AsyncIterator[str]",
        holdback_chars: int = DEFAULT_HOLDBACK_CHARS,
        principal: str | None = None,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ) -> "GuardedStream":
        """Wrap a token stream so it is scanned while it is still arriving.

            guarded = pipeline.guard_stream(model_stream)
            async for chunk in guarded:
                await websocket.send(chunk)
            if guarded.blocked:
                await websocket.replace(guarded.replacement_text)

        The iteration ends silently on a block rather than yielding the
        refusal as a final chunk, because appending it would leave the
        offending text above it on screen. Replacing the message is the
        caller's job and there is no way for this to do it for them.
        """
        return GuardedStream(
            self, source, holdback_chars=holdback_chars, principal=principal,
            min_chunk_chars=min_chunk_chars,
        )

    async def _scan_tool_result(
        self, action: str, output: object, session_id: str | None, principal: str | None,
    ) -> None:
        text = "\n".join(t for t in _strings_in(output) if t.strip())
        if not text:
            return
        # The same path a RAG chunk takes: lexical scan, detectors, audit,
        # metrics, and the session charged if there is one — a tool ran
        # because the user asked for it, so its result is theirs.
        scan = await self.pre_process_external(
            text, source_id=f"tool:{action}", session_id=session_id, principal=principal,
        )
        if scan.blocked:
            self.metrics.increment(
                "blocks_total" if self.enforcement == "enforce" else "would_block_total",
                stage="tool_result", reason="risk_score",
            )
            await self._audit("tool_result", {
                "session_id": session_id, "principal": principal, "action": action,
                "risk_score": scan.risk_score, "matched_patterns": scan.matched_patterns,
                "blocked": self.enforcement == "enforce", "would_block": True,
            })
            if self.enforcement == "enforce":
                raise ToolResultBlocked(action, scan, output)

    # -- Degradation reporting --------------------------------------------

    def _adopt_reporting(self, *guards: object) -> None:
        """Point a guard's degradation callback at this pipeline.

        Only the callback: the guard's own FailurePolicy is untouched. If
        the same guard instance is shared between two pipelines the later
        one wins, which is the price of guards being independently usable.
        """
        for guard in guards:
            backend = getattr(guard, "_resilience", None) or getattr(guard, "resilience", None)
            if isinstance(backend, ResilientBackend) and backend is not self._resilience:
                backend.with_callback(self._on_degraded)

    async def _on_degraded(self, event: Degradation) -> None:
        """Called by ResilientBackend whenever a guard did not run.

        Two destinations, on purpose. The audit event is for whoever reads
        the logs afterwards; the per-call bucket is for the caller handling
        this request right now, who is the only one in a position to decide
        whether to serve a reply that was not fully checked.
        """
        bucket = _current_degradations.get()
        if bucket is not None:
            bucket.append(event)
        self.metrics.increment(
            "degradations_total", operation=event.operation, decision=event.decision,
        )
        if event.operation == AUDIT:
            # The audit logger is what just failed. ResilientBackend has
            # already emitted a warning through `logging`; trying to record
            # the failure through the thing that failed would recurse.
            return
        await self._audit("backend_degraded", event.as_dict())

    @contextlib.contextmanager
    def _collecting_degradations(self):
        """Per-call bucket, held in a ContextVar so concurrent requests
        sharing one pipeline don't collect each other's failures."""
        bucket: list[Degradation] = []
        token = _current_degradations.set(bucket)
        try:
            yield bucket
        finally:
            _current_degradations.reset(token)

    async def _audit(self, event_type: str, data: dict) -> None:
        """Write an audit event under the failure policy.

        Defaults to open: losing a log line should not refuse the request
        the line was describing. Set `FailurePolicy(audit="closed")` if your
        compliance position is that an unlogged request must not happen.
        """
        stamped = {"schema_version": AUDIT_SCHEMA_VERSION, **data}
        await self._resilience.run(AUDIT, lambda: self.audit.log(event_type, stamped))

    # -- Lifecycle -----------------------------------------------------

    async def __aenter__(self) -> "SecurityPipeline":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the process pool and let the state backend release any
        connection it owns. Call this on application shutdown; don't skip
        it, or worker processes/connections leak."""
        if self._owns_process_executor:
            self.process_executor.shutdown(wait=True, cancel_futures=False)
        if self.state_backend is not None:
            await self.state_backend.aclose()

    # -- Identity ----------------------------------------------------------

    def _require_principal(self, principal: str | None, where: str) -> None:
        if self.session_identity == "authenticated" and principal is None:
            raise ValueError(
                f"{where} requires principal= because this pipeline was built with "
                "session_identity='authenticated'. Pass the authenticated end-user "
                "identifier from your gateway."
            )

    def session_key(self, session_id: str, principal: str | None = None) -> str:
        """The key session state is actually stored under.

        With a principal, the caller's session id stops being the whole
        identity: guessing or reusing someone else's produces a different
        key, so it inherits none of their budget or accumulated risk. The
        principal is not verified here — it is verified by whatever issued
        it — but the binding becomes structural instead of optional.
        """
        if principal is None:
            return session_id
        return f"{principal}||{session_id}"

    # -- Direct input (the user's own message) -----------------------------

    async def pre_process(
        self,
        user_input: str,
        session_id: str | None = None,
        principal: str | None = None,
        actor_id: str | None = None,
    ) -> PreProcessResult:
        """Scan a user message.

        `principal` is the authenticated end user, when your gateway can
        supply one; it binds the session state to that identity. `actor_id`
        is the coarser fallback — an account, API key, tenant or source
        address — against which risk accumulates even when sessions rotate.
        It defaults to the principal when one is given.
        """
        with self._collecting_degradations() as degraded, self.tracer.span(
            "pre_process", enforcement=self.enforcement,
        ) as span:
            with self.metrics.timed("scan_duration_seconds", stage="input"):
                try:
                    result = await self._pre_process(
                        user_input, session_id=session_id, principal=principal,
                        actor_id=actor_id, degraded=degraded,
                    )
                except RateLimitExceeded:
                    # Under enforcement this leaves as an exception, so it
                    # would otherwise never reach the counters and the
                    # requests_total series would silently under-count the
                    # traffic it claims to describe.
                    self.metrics.increment(
                        "requests_total", stage="input", outcome="rate_limited",
                        enforcement=self.enforcement,
                    )
                    self.metrics.increment("rate_limited_total", stage="input")
                    self.metrics.increment(
                        "blocks_total", stage="input", reason="rate_limit",
                    )
                    span.set_attribute("outcome", "rate_limited")
                    raise
            span.set_attribute("outcome", "blocked" if result.blocked else "allowed")
            span.set_attribute("risk_score", result.sanitized.risk_score)
            span.set_attribute("combined_risk_score", result.combined_risk_score)
            span.set_attribute("matched_patterns", result.sanitized.matched_patterns)
            span.set_attribute("degraded", [d.operation for d in result.degraded])
        self._record_input_metrics(result)
        return result

    def _record_input_metrics(self, result: PreProcessResult) -> None:
        metrics = self.metrics
        metrics.increment(
            "requests_total",
            stage="input",
            outcome="blocked" if result.blocked else "allowed",
            enforcement=self.enforcement,
        )
        # Both scores, labelled: choosing a threshold against the merged
        # score when the lexical one is what you are tuning is a good way
        # to pick the wrong number.
        metrics.observe("risk_score", result.sanitized.risk_score, stage="input", kind="lexical")
        metrics.observe("risk_score", result.combined_risk_score, stage="input", kind="combined")
        for pattern in result.sanitized.matched_patterns:
            metrics.increment("findings_total", stage="input", kind="pattern", category=pattern)
        if result.rate_limited:
            metrics.increment("rate_limited_total", stage="input")
        if result.blocked:
            metrics.increment("blocks_total", stage="input", reason=self._input_reason(result))
        elif result.would_block:
            # The shadow-mode number: what enforcement would have done.
            # Comparing the two series is the whole point of shadow mode.
            metrics.increment(
                "would_block_total", stage="input", reason=self._input_reason(result),
            )
        if result.detectors is not None:
            for detector in result.detectors.results:
                mode = "enforcing" if detector.detector in self.detectors.enforcing else "advisory"
                metrics.observe(
                    "detector_score", detector.score, detector=detector.detector, mode=mode,
                )
            for name in result.detectors.errors:
                metrics.increment("detector_errors_total", detector=name)

    def _input_reason(self, result: PreProcessResult) -> str:
        """Which signal decided. Low cardinality by construction: four
        possible values, so it is safe as a metric label."""
        if result.rate_limited:
            return "rate_limit"
        if result.sanitized.oversized:
            return "oversized"
        if result.combined_risk_score >= self.input_risk_threshold:
            return "risk_score"
        # Nothing about this turn on its own; the session accumulated it.
        return "session_risk"

    async def _pre_process(
        self,
        user_input: str,
        session_id: str | None,
        principal: str | None,
        actor_id: str | None,
        degraded: list[Degradation],
    ) -> PreProcessResult:
        self._require_principal(principal, "pre_process")
        actor_id = actor_id or principal
        limits = self._limits(principal)
        scoped_session = (
            self.session_key(session_id, principal) if session_id is not None else None
        )
        loop = asyncio.get_running_loop()

        # Either an executor Future or a plain coroutine, depending on the
        # branch below; the only thing the caller does with it is await it.
        scan_awaitable: Awaitable[SanitizationResult]

        # The scan itself is fast for typical chat-message sizes, so it's
        # run inline by default (see module docstring for why offloading a
        # single small task to a process pool would add latency, not
        # remove it). Only very large pasted inputs are offloaded, where
        # the regex work is large enough to be worth the IPC cost.
        if len(user_input) >= self.large_input_offload_threshold_chars:
            scan_awaitable = loop.run_in_executor(
                self.process_executor,
                self.sanitizer.scan_text,
                user_input,
                self.input_risk_threshold,
                "USER_DATA",
                session_id and f"user_message:{session_id}",
            )
        else:
            async def _inline_scan() -> SanitizationResult:
                return self.sanitizer.scan_text(
                    user_input,
                    threshold=self.input_risk_threshold,
                    tag="USER_DATA",
                    source_id=session_id and f"user_message:{session_id}",
                )
            scan_awaitable = _inline_scan()

        # A network-backed judge adds its full latency to every turn if it
        # is awaited in sequence, so it starts now and is collected after
        # the lexical scan rather than before it.
        detector_task = (
            asyncio.ensure_future(self.detectors.score(user_input))
            if self.detectors
            else None
        )

        # Redis budget check is independent I/O: run it concurrently with
        # the scan rather than sequentially after it.
        rate_limited = False
        if scoped_session is not None:
            if self.enforcement == "shadow":
                # The counter still has to be incremented — a shadow
                # deployment that does not count is not measuring the same
                # system — but going over budget is recorded, not raised.
                scan_outcome, limit_outcome = await asyncio.gather(
                    scan_awaitable,
                    self.rate_limiter.check_request(scoped_session, limits=limits),
                    return_exceptions=True,
                )
                if isinstance(scan_outcome, BaseException):
                    raise scan_outcome
                result = scan_outcome
                rate_limited = isinstance(limit_outcome, RateLimitExceeded)
                if isinstance(limit_outcome, BaseException) and not rate_limited:
                    raise limit_outcome
            else:
                result, _ = await asyncio.gather(
                    scan_awaitable, self.rate_limiter.check_request(scoped_session, limits=limits)
                )
        else:
            result = await scan_awaitable

        if detector_task is not None:
            with self.tracer.span("detectors") as det_span:
                ensemble = await detector_task
                det_span.set_attribute("enforcing_score", ensemble.enforcing_score)
                det_span.set_attribute("errors", list(ensemble.errors))
        else:
            ensemble = None
        combined_risk = (
            self.detectors.combine(result.risk_score, ensemble)
            if ensemble is not None
            else result.risk_score
        )
        # Recomputed rather than trusting `result.blocked`: the sanitizer
        # only saw the lexical half of the evidence.
        detector_blocked = combined_risk >= self.input_risk_threshold

        cumulative = None
        session_flagged = False
        if scoped_session is not None:
            # One call, both answers: the store decides the flag while it
            # applies the risk, so asking for it separately was a second
            # (and with an actor, third) round trip for nothing.
            cumulative, session_flagged = await self.rate_limiter.record_turn_risk_and_check(
                scoped_session, combined_risk, actor_id=actor_id, limits=limits,
            )

        await self._audit("input_scan", {
            "session_id": session_id,
            "principal": principal,
            "actor_id": actor_id,
            "enforcement": self.enforcement,
            "risk_score": result.risk_score,
            "combined_risk_score": combined_risk,
            "detectors": ensemble.as_audit() if ensemble is not None else None,
            "cumulative_session_risk": cumulative,
            "session_flagged": session_flagged,
            "matched_patterns": result.matched_patterns,
            "matched_languages": list(result.matched_languages),
            "decoded_payload_hits": bool(result.decoded_payload_hits),
            "blocked": result.blocked,
            "degraded": [d.operation for d in degraded],
        })

        # A session flagged for cumulative, gradually-escalating risk is
        # blocked even if this specific turn looked fine on its own — this
        # is what catches multi-turn jailbreak build-up.
        would_block = detector_blocked or session_flagged or rate_limited
        return PreProcessResult(
            sanitized=result,
            blocked=would_block if self.enforcement == "enforce" else False,
            would_block=would_block,
            rate_limited=rate_limited,
            detectors=ensemble,
            combined_risk_score=combined_risk,
            degraded=tuple(degraded),
        )

    # -- External / retrieved content (indirect prompt injection) ----------

    async def pre_process_external(
        self,
        content: str,
        source_id: str,
        threshold: float | None = None,
        session_id: str | None = None,
        principal: str | None = None,
        actor_id: str | None = None,
    ) -> SanitizationResult:
        """Scan a single piece of content the agent did NOT receive
        directly from the user — a web page, a RAG chunk, a tool output.
        This is the indirect-injection surface. For scanning several
        chunks at once, prefer pre_process_external_batch, which actually
        parallelizes across CPU cores.

        Passing `session_id` feeds the resulting risk into that session's
        cumulative total, the way pre_process and pre_process_media do. It
        is opt-in rather than automatic because the two readings are both
        defensible and the caller is the one who knows which applies: a
        poisoned page the user never chose is not evidence about the user,
        while a user steering retrieval at a document they planted is.
        Leaving it off keeps the previous behaviour, where external content
        is scanned and audited but never charged to anyone.
        """
        with self.tracer.span("pre_process_external") as span:
            result = await self._pre_process_external(
                content, source_id, threshold, session_id, principal, actor_id,
            )
            span.set_attribute("outcome", "blocked" if result.blocked else "allowed")
            span.set_attribute("risk_score", result.risk_score)
            return result

    async def _pre_process_external(
        self, content, source_id, threshold, session_id, principal, actor_id,
    ) -> SanitizationResult:
        effective_threshold = threshold if threshold is not None else self.input_risk_threshold
        detector_task = (
            asyncio.ensure_future(self.detectors.score(content)) if self.detectors else None
        )
        result = self.sanitizer.scan_text(
            content,
            threshold=effective_threshold,
            tag="EXTERNAL_CONTENT",
            source_id=source_id,
        )
        # Indirect injection is where a semantic detector earns its keep:
        # a poisoned document is written to be read, so it is paraphrased
        # prose rather than a phrase from a list.
        ensemble = await detector_task if detector_task is not None else None
        if ensemble is not None:
            result = self._merge_detectors(result, ensemble, effective_threshold)

        self.metrics.increment(
            "requests_total",
            stage="external",
            outcome="blocked" if result.blocked else "allowed",
            enforcement=self.enforcement,
        )
        self.metrics.observe("risk_score", result.risk_score, stage="external", kind="combined")
        for pattern in result.matched_patterns:
            self.metrics.increment(
                "findings_total", stage="external", kind="pattern", category=pattern,
            )
        if result.blocked:
            self.metrics.increment("blocks_total", stage="external", reason="risk_score")

        await self._audit("external_content_scan", {
            "source_id": source_id,
            "risk_score": result.risk_score,
            "detectors": ensemble.as_audit() if ensemble is not None else None,
            "matched_patterns": result.matched_patterns,
            "matched_languages": list(result.matched_languages),
            "decoded_payload_hits": bool(result.decoded_payload_hits),
            "blocked": result.blocked,
        })
        await self._record_external_risk(
            [result], session_id=session_id, principal=principal, actor_id=actor_id,
        )
        return result

    def _merge_detectors(
        self,
        result: SanitizationResult,
        ensemble: EnsembleResult,
        threshold: float,
    ) -> SanitizationResult:
        """Fold enforcing detector scores into an external-content verdict.

        External content has no PreProcessResult to carry the two scores
        separately, so here the merged score does replace the lexical one —
        and `detector:<name>` is appended to `matched_patterns` so the
        record still says which signal fired. The full breakdown is in the
        audit event either way.
        """
        combined = self.detectors.combine(result.risk_score, ensemble)
        if combined == result.risk_score:
            return result
        labels = [f"detector:{r.detector}" for r in ensemble.results]
        return dataclasses.replace(
            result,
            risk_score=combined,
            matched_patterns=[*result.matched_patterns, *labels],
            blocked=combined >= threshold,
        )

    async def _record_external_risk(
        self,
        results: list[SanitizationResult],
        session_id: str | None,
        principal: str | None,
        actor_id: str | None,
    ) -> None:
        """Charge external-content risk to a session, when the caller asked
        for it. The highest-scoring chunk is charged rather than the sum, so
        a wide retrieval doesn't flag a session for being wide."""
        if session_id is None:
            return
        worst = max((r.risk_score for r in results), default=0.0)
        if worst <= 0:
            return
        await self.rate_limiter.record_turn_risk(
            self.session_key(session_id, principal),
            worst,
            actor_id=actor_id or principal,
        )

    async def pre_process_external_batch(
        self,
        chunks: list[tuple[str, str]],
        threshold: float | None = None,
        session_id: str | None = None,
        principal: str | None = None,
        actor_id: str | None = None,
    ) -> list[SanitizationResult]:
        """Scan multiple retrieved chunks (e.g. several RAG search results,
        or several pages fetched by a browsing agent). This is the actual
        parallelization win in this pipeline: below
        `external_scan_parallel_min_chunks` chunks it isn't worth paying
        process-pool overhead, so they run inline; at or above it, each
        chunk's regex scan runs in a separate OS process, genuinely using
        multiple CPU cores at once instead of taking turns on the GIL."""
        if not chunks:
            return []

        effective_threshold = threshold if threshold is not None else self.input_risk_threshold
        loop = asyncio.get_running_loop()
        results = (
            [
                self.sanitizer.scan_text(
                    content, threshold=effective_threshold, tag="EXTERNAL_CONTENT", source_id=source_id,
                )
                for content, source_id in chunks
            ]
            if len(chunks) < self.external_scan_parallel_min_chunks
            else await asyncio.gather(*[
                loop.run_in_executor(
                    self.process_executor,
                    self.sanitizer.scan_text,
                    content,
                    effective_threshold,
                    "EXTERNAL_CONTENT",
                    source_id,
                )
                for content, source_id in chunks
            ])
        )

        # Audit log writes are independent I/O: fire them concurrently.
        await asyncio.gather(*[
            self._audit("external_content_scan", {
                "source_id": result.source_id,
                "risk_score": result.risk_score,
                "matched_patterns": result.matched_patterns,
                "matched_languages": list(result.matched_languages),
                "decoded_payload_hits": bool(result.decoded_payload_hits),
                "blocked": result.blocked,
            })
            for result in results
        ])

        await self._record_external_risk(
            list(results), session_id=session_id, principal=principal, actor_id=actor_id,
        )

        return list(results)

    # -- Output ---------------------------------------------------------

    async def post_process(self, model_output: str, principal: str | None = None) -> PostProcessResult:
        with self._collecting_degradations() as degraded, self.tracer.span(
            "post_process", enforcement=self.enforcement,
        ) as span:
            with self.metrics.timed("scan_duration_seconds", stage="output"):
                forbidden = await self._forbidden_literals(principal)
                result = await self._post_process(model_output, forbidden)
            span.set_attribute("outcome", "blocked" if result.blocked else "allowed")
            span.set_attribute("secret_categories", list(result.scan.secret_findings))
            span.set_attribute("system_prompt_overlap_score", result.scan.system_prompt_overlap_score)
        result = dataclasses.replace(result, degraded=tuple(degraded))
        self._record_output_metrics(result, streamed=False)
        return result

    def _record_output_metrics(self, result: PostProcessResult, streamed: bool) -> None:
        metrics = self.metrics
        stage = "output_stream" if streamed else "output"
        metrics.increment(
            "requests_total",
            stage=stage,
            outcome="blocked" if result.blocked else "allowed",
            enforcement=self.enforcement,
        )
        metrics.observe(
            "risk_score", result.scan.system_prompt_overlap_score,
            stage=stage, kind="system_prompt_overlap",
        )
        for kind, findings in (
            ("secret", result.scan.secret_findings), ("pii", result.scan.pii_findings),
        ):
            for category in findings:
                metrics.increment("findings_total", stage=stage, kind=kind, category=category)
        reason = self._output_reason(result)
        if result.blocked:
            metrics.increment("blocks_total", stage=stage, reason=reason)
        elif result.would_block:
            metrics.increment("would_block_total", stage=stage, reason=reason)

    @staticmethod
    def _output_reason(result: PostProcessResult) -> str:
        if result.scan.oversized:
            return "oversized"
        if result.scan.secret_findings:
            return "secret"
        if result.exfil is not None and result.exfil.blocked:
            return "exfil"
        if result.scan.system_prompt_overlap_score > 0:
            return "system_prompt_overlap"
        return "none"

    async def _forbidden_literals(self, principal: str | None) -> tuple[str, ...]:
        if self.foreign_identifiers is None or principal is None:
            return ()
        raw = self.foreign_identifiers(principal)
        if inspect.isawaitable(raw):
            raw = await raw
        return clean_forbidden_literals(raw or ())

    async def _post_process(
        self, model_output: str, forbidden: tuple[str, ...] = (),
    ) -> PostProcessResult:
        loop = asyncio.get_running_loop()
        # Both scans are pure CPU (regex); offload only if the text is large
        # enough for that to matter (same rationale as pre_process). They go
        # over together in one call: the exfil scan consumes the output
        # guard's redacted text, so splitting them would mean two IPC round
        # trips and a serialization of the same string twice.
        if len(model_output) >= self.large_input_offload_threshold_chars:
            result, exfil = await loop.run_in_executor(
                self.process_executor,
                _scan_output,
                self.output_guard,
                self.exfil_guard if self.scan_output_for_exfil else None,
                model_output,
                self.system_prompt,
                self.output_overlap_threshold,
                forbidden,
            )
        else:
            result, exfil = _scan_output(
                self.output_guard,
                self.exfil_guard if self.scan_output_for_exfil else None,
                model_output,
                self.system_prompt,
                self.output_overlap_threshold,
                forbidden,
            )
        blocked = result.blocked or (exfil is not None and exfil.blocked)

        await self._audit("output_scan", {
            "oversized": result.oversized,
            "secret_categories": list(result.secret_findings.keys()),
            "pii_categories": list(result.pii_findings.keys()),
            "system_prompt_overlap_score": result.system_prompt_overlap_score,
            "exfil_channels": [f.channel for f in exfil.suspicious_findings] if exfil else [],
            "exfil_reasons": sorted(exfil.reasons) if exfil else [],
            "exfil_blocked": bool(exfil and exfil.blocked),
            "blocked": blocked,
        })

        would_block = blocked
        if self.enforcement == "shadow":
            # Observed, not applied: the caller gets exactly what the model
            # produced, including anything that would have been redacted.
            return PostProcessResult(
                scan=result, safe_text=model_output, blocked=False,
                exfil=exfil, would_block=would_block,
            )

        safe_text = (
            "[Response blocked by the security layer: possible credential "
            "or system-prompt leak detected.]"
        )
        if result.oversized:
            # Never scanned, so nothing is known about it. Withheld rather
            # than passed through, because "too big to check" and "checked
            # and clean" must not produce the same reply.
            safe_text = result.redacted_text
        if not blocked:
            # Neutralized, not merely redacted: any click-required link with
            # a payload shape has had its destination stripped even though
            # it was not severe enough to block the reply.
            safe_text = exfil.neutralized_text if exfil else result.redacted_text
        elif exfil is not None and exfil.blocked and not result.blocked:
            safe_text = (
                "[Response blocked by the security layer: it contained a URL that "
                "would have transmitted data to a third party when rendered.]"
            )

        return PostProcessResult(
            scan=result,
            safe_text=safe_text,
            blocked=blocked,
            exfil=exfil,
            would_block=would_block,
        )

    # -- Ingestion (index-time) -------------------------------------------

    async def ingest_document(
        self, content: str, *, document_id: str, source_id: str, trust: str | None = None,
    ) -> IngestVerdict:
        """Scan a document on its way into a knowledge base and record the
        verdict. See services/ingest_guard.py for why this is a separate
        entry point from pre_process_external rather than the same scan run
        earlier."""
        verdict = await self.ingest_guard.ingest(
            content, document_id=document_id, source_id=source_id, trust=trust,
        )
        await self._audit("ingest", {
            "document_id": verdict.document_id,
            "source_id": verdict.source_id,
            "trust": verdict.trust,
            "decision": verdict.decision,
            "risk_score": verdict.risk_score,
            "reasons": list(verdict.reasons),
        })
        return verdict

    async def ingest_batch(
        self, documents: list[tuple[str, str, str]], trust: str | None = None,
    ) -> list[IngestVerdict]:
        """Evaluate many documents at once, as (content, document_id,
        source_id) triples.

        Scanning is pure and CPU-bound, so it goes across the process pool
        on the same terms as pre_process_external_batch; recording the
        verdicts is I/O and happens afterwards, concurrently.
        """
        if not documents:
            return []
        loop = asyncio.get_running_loop()

        if len(documents) < self.external_scan_parallel_min_chunks:
            verdicts = [
                self.ingest_guard.evaluate(
                    content, document_id=document_id, source_id=source_id, trust=trust,
                )
                for content, document_id, source_id in documents
            ]
        else:
            # The guard sent to the worker processes is a copy stripped of
            # everything that talks to a backend: the provenance store owns
            # a connection, and the ResilientBackend holds a callback bound
            # to this pipeline, so neither is picklable and neither has any
            # business being duplicated per worker. `evaluate` is pure and
            # needs neither. Recording and failure handling stay in this
            # process, where the connection and the breakers live.
            scanner = dataclasses.replace(
                self.ingest_guard, provenance_store=None, resilience=None, failure_policy=None,
            )
            verdicts = await asyncio.gather(*[
                loop.run_in_executor(
                    self.process_executor,
                    functools.partial(
                        scanner.evaluate,
                        content, document_id=document_id, source_id=source_id, trust=trust,
                    ),
                )
                for content, document_id, source_id in documents
            ])

        if self.ingest_guard.provenance_store is not None:
            await asyncio.gather(*[
                self.ingest_guard.provenance_store.record(ProvenanceRecord(
                    document_id=v.document_id, content_hash=v.content_hash,
                    source_id=v.source_id, trust=v.trust,
                    decision=v.decision, risk_score=v.risk_score,
                ))
                for v in verdicts
            ])
        await asyncio.gather(*[
            self._audit("ingest", {
                "document_id": v.document_id, "source_id": v.source_id,
                "trust": v.trust, "decision": v.decision,
                "risk_score": v.risk_score, "reasons": list(v.reasons),
            })
            for v in verdicts
        ])
        return list(verdicts)

    async def verify_retrieved(self, content: str, *, document_id: str) -> RetrievalVerdict:
        """Check a retrieved chunk against what was recorded when it was
        indexed. Catches content modified in the store after ingest, which
        no amount of scanning at ingest time can."""
        verdict = await self.ingest_guard.verify_retrieved(content, document_id=document_id)
        if not verdict.trusted:
            await self._audit("retrieval_provenance", {
                "document_id": document_id, "reason": verdict.reason,
                "tampered": verdict.tampered,
            })
        return verdict

    # -- Non-text input ----------------------------------------------------

    async def pre_process_media(
        self,
        payload: bytes,
        media_type: str = "application/octet-stream",
        source_id: str = "media",
        session_id: str | None = None,
        principal: str | None = None,
        actor_id: str | None = None,
    ) -> MediaScanResult:
        """Scan a non-text payload by running the configured extractors and
        putting whatever text they recover through the ordinary sanitizer.

        With the default extractors this sees metadata and embedded
        strings, not pixels. Supply an OCR or vision extractor to cover
        rendered text; see services/media_guard.py.
        """
        self._require_principal(principal, "pre_process_media")
        actor_id = actor_id or principal
        with self.tracer.span("pre_process_media", media_type=media_type) as span:
            result = await self.media_scanner.scan(payload, media_type=media_type, source_id=source_id)
            span.set_attribute("extractors", [e.extractor for e in result.extractions])

        # Whatever the extractors recovered is text the model will read, so
        # it is as much a detector's business as a typed message is — more,
        # if anything: a document written to be read is paraphrased prose,
        # which is exactly what the lexical scan misses. One call over the
        # concatenated extractions, mirroring one call per message.
        ensemble = None
        recovered = "\n\n".join(e.text for e in result.extractions if e.text.strip())
        if self.detectors and recovered:
            ensemble = await self.detectors.score(recovered)
            result.detectors = ensemble
            result.combined_risk_score = self.detectors.combine(result.risk_score, ensemble)
            result.blocked = result.combined_risk_score >= self.media_scanner.threshold

        if session_id is not None and result.combined_risk_score > 0:
            await self.rate_limiter.record_turn_risk(
                self.session_key(session_id, principal), result.combined_risk_score,
                actor_id=actor_id,
            )
        await self._audit("media_scan", {
            "session_id": session_id,
            "principal": principal,
            "source_id": source_id,
            "media_type": media_type,
            "extractors": [e.extractor for e in result.extractions],
            "extractor_errors": result.extractor_errors,
            "risk_score": result.risk_score,
            "combined_risk_score": result.combined_risk_score,
            "detectors": ensemble.as_audit() if ensemble is not None else None,
            "matched_patterns": result.matched_patterns,
            "blocked": result.blocked,
        })
        return result

    # -- Tool calling / budget -------------------------------------------

    async def check_tool_call_budget(
        self, session_id: str, principal: str | None = None,
    ) -> None:
        """Raises RateLimitExceeded if the session has made too many tool
        calls within the current window — mitigates DoS/cost abuse via
        excessive tool usage, independent of whether each individual call
        was in-scope."""
        await self.rate_limiter.check_tool_call(
            self.session_key(session_id, principal), limits=self._limits(principal),
        )

    async def authorized_tool_call(
        self,
        token,
        action: str,
        func: Callable,
        *args,
        session_id: str | None = None,
        principal: str | None = None,
        **kwargs,
    ):
        """Run a tool call behind the scope guard.

        `principal` is presented to the capability token: a token issued
        with a subject can only be spent by that subject, which is what
        makes a leaked token useless to whoever finds it.

        With `scan_tool_results` (the default) the tool's return value is
        scanned as external content before it is handed back, and raises
        `ToolResultBlocked` if it looks like an injection: the arguments
        were checked on the way out, and what comes back is what the model
        reads next. Callers should still put the result through
        `wrap_as_data` when building the prompt — that is the structural
        defence; this is the detection in front of it.
        """
        self._require_principal(principal, "authorized_tool_call")
        with self.tracer.span("tool_call", action=action) as span:
            return await self._authorized_tool_call(
                token, action, func, *args, session_id=session_id, principal=principal,
                _span=span, **kwargs,
            )

    async def _authorized_tool_call(
        self, token, action, func, *args, session_id, principal, _span, **kwargs,
    ):
        if session_id is not None:
            await self.check_tool_call_budget(session_id, principal)
        if self.scan_tool_call_arguments:
            # Checked before the scope check runs, because an in-scope
            # action with an attacker-chosen destination is the whole
            # problem: `http_get` may be perfectly authorized.
            findings = self.exfil_guard.scan_values([args, kwargs])
            if findings and self.enforcement == "shadow":
                await self._audit("tool_call", {
                    "session_id": session_id, "principal": principal,
                    "action": action, "status": "would_deny",
                    "reason": "exfiltration_in_arguments",
                    "exfil_reasons": sorted({r for f in findings for r in f.reasons}),
                })
            elif findings:
                await self._audit("tool_call", {
                    "session_id": session_id, "principal": principal,
                    "action": action, "status": "denied",
                    "reason": "exfiltration_in_arguments",
                    "exfil_reasons": sorted({r for f in findings for r in f.reasons}),
                })
                raise ExfilAttemptBlocked(findings)
        try:
            output = await self.scope_guard.guarded_call(
                token, action, func, *args, subject=principal, **kwargs,
            )
            self.metrics.increment(
                "requests_total", stage="tool_call", outcome="allowed",
                enforcement=self.enforcement,
            )
            await self._audit("tool_call", {
                "session_id": session_id, "principal": principal,
                "action": action, "status": "allowed",
                # Which signing key this token was issued under. This is how
                # a rotation is finished safely: the old key can be retired
                # once no spent token has named it for longer than the
                # longest TTL you issue, and that is a query against this
                # field rather than a guess.
                "token_key_id": getattr(token, "key_id", None),
            })
        except Exception as exc:
            # The exception TYPE is the reason and is low cardinality; its
            # message is not, and may quote the input, so it stays in the
            # audit log where identifiers already live.
            self.metrics.increment(
                "requests_total", stage="tool_call", outcome="denied",
                enforcement=self.enforcement,
            )
            self.metrics.increment(
                "blocks_total", stage="tool_call", reason=type(exc).__name__,
            )
            if isinstance(exc, RateLimitExceeded):
                self.metrics.increment("rate_limited_total", stage="tool_call")
            await self._audit("tool_call", {
                "session_id": session_id, "principal": principal,
                "action": action, "status": "denied", "reason": str(exc),
                "token_key_id": getattr(token, "key_id", None),
            })
            raise

        # Outside the try: the CALL was allowed and is audited as such above.
        # What is being judged now is the result, which is a different event
        # with its own audit record, its own metric, and its own exception.
        if self.scan_tool_results:
            with self.tracer.span("tool_result", action=action):
                await self._scan_tool_result(action, output, session_id, principal)
        _span.set_attribute("outcome", "allowed")
        return output
