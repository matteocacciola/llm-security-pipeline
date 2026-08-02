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
import dataclasses
import logging
import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

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
    OutputGuard,
    OutputScanResult,
    SessionRateLimiter,
    SessionLimits,
    RateLimitExceeded,
)
from .sessions.stores import ProvenanceRecord

logger = logging.getLogger(__name__)
from .state_backend import StateBackend, RedisStateBackend

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


@dataclass
class PostProcessResult:
    scan: OutputScanResult
    safe_text: str
    blocked: bool
    exfil: ExfilScanResult | None = None
    # In shadow mode `blocked` is always False while `would_block` records
    # what enforcement would have done. Equal to `blocked` otherwise.
    would_block: bool = False


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
        await self._redis.xadd(self._stream_name, payload, maxlen=self._maxlen, approximate=True)


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
) -> tuple[OutputScanResult, ExfilScanResult | None]:
    result = output_guard.scan(
        text, system_prompt=system_prompt, overlap_threshold=overlap_threshold,
    )
    # The side-channel scan runs over the REDACTED text: a secret the output
    # guard has already replaced can no longer be smuggled out in a URL, and
    # scanning the original would re-flag it.
    exfil = exfil_guard.scan(result.redacted_text) if exfil_guard is not None else None
    return result, exfil


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SecurityPipeline:
    def __init__(
        self,
        system_prompt: str | None = None,
        input_risk_threshold: float = 0.6,
        output_overlap_threshold: float = 0.35,
        scope_guard: ScopeGuard | None = None,
        sanitizer: Sanitizer | None = None,
        output_guard: OutputGuard | None = None,
        exfil_guard: ExfilGuard | None = None,
        exfil_allowed_hosts: list[str] | None = None,
        scan_output_for_exfil: bool = True,
        scan_tool_call_arguments: bool = True,
        ingest_guard: IngestGuard | None = None,
        media_scanner: MediaScanner | None = None,
        enforcement: str = "enforce",
        session_identity: str | None = None,
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

        if scope_guard is None:
            scope_guard = ScopeGuard(
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
            )
        self.rate_limiter = rate_limiter

        self.sanitizer = sanitizer or Sanitizer(
            pattern_config_path=injection_config_path,
            include_default_patterns=include_default_injection_patterns,
        )

        self.output_guard = output_guard or OutputGuard(
            pattern_config_path=pii_config_path,
            include_default_patterns=include_default_pii_patterns,
        )

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
        self.ingest_guard = ingest_guard or IngestGuard(
            sanitizer=self.sanitizer,
            exfil_guard=self.exfil_guard,
            provenance_store=getattr(state_backend, "provenance_store", None),
        )

        # Non-text inputs. The default scanner reads printable strings only;
        # plug in OCR or a vision model via MediaScanner(extractors=[...])
        # to cover text rendered as pixels. See media_guard.py for why no
        # OCR engine is bundled.
        self.media_scanner = media_scanner or MediaScanner(sanitizer=self.sanitizer)

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
        self._require_principal(principal, "pre_process")
        actor_id = actor_id or principal
        scoped_session = (
            self.session_key(session_id, principal) if session_id is not None else None
        )
        loop = asyncio.get_running_loop()

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
            async def _inline_scan():
                return self.sanitizer.scan_text(
                    user_input,
                    threshold=self.input_risk_threshold,
                    tag="USER_DATA",
                    source_id=session_id and f"user_message:{session_id}",
                )
            scan_awaitable = _inline_scan()

        # Redis budget check is independent I/O: run it concurrently with
        # the scan rather than sequentially after it.
        rate_limited = False
        if scoped_session is not None:
            if self.enforcement == "shadow":
                # The counter still has to be incremented — a shadow
                # deployment that does not count is not measuring the same
                # system — but going over budget is recorded, not raised.
                results = await asyncio.gather(
                    scan_awaitable,
                    self.rate_limiter.check_request(scoped_session),
                    return_exceptions=True,
                )
                result, limit_outcome = results
                if isinstance(result, BaseException):
                    raise result
                rate_limited = isinstance(limit_outcome, RateLimitExceeded)
                if isinstance(limit_outcome, BaseException) and not rate_limited:
                    raise limit_outcome
            else:
                result, _ = await asyncio.gather(
                    scan_awaitable, self.rate_limiter.check_request(scoped_session)
                )
        else:
            result = await scan_awaitable

        cumulative = None
        session_flagged = False
        if scoped_session is not None:
            cumulative = await self.rate_limiter.record_turn_risk(
                scoped_session, result.risk_score, actor_id=actor_id,
            )
            session_flagged = await self.rate_limiter.is_session_flagged(
                scoped_session, actor_id=actor_id,
            )

        await self.audit.log("input_scan", {
            "session_id": session_id,
            "principal": principal,
            "actor_id": actor_id,
            "enforcement": self.enforcement,
            "risk_score": result.risk_score,
            "cumulative_session_risk": cumulative,
            "session_flagged": session_flagged,
            "matched_patterns": result.matched_patterns,
            "matched_languages": list(result.matched_languages),
            "decoded_payload_hits": bool(result.decoded_payload_hits),
            "blocked": result.blocked,
        })

        # A session flagged for cumulative, gradually-escalating risk is
        # blocked even if this specific turn looked fine on its own — this
        # is what catches multi-turn jailbreak build-up.
        would_block = result.blocked or session_flagged or rate_limited
        return PreProcessResult(
            sanitized=result,
            blocked=would_block if self.enforcement == "enforce" else False,
            would_block=would_block,
            rate_limited=rate_limited,
        )

    # -- External / retrieved content (indirect prompt injection) ----------

    async def pre_process_external(
        self, content: str, source_id: str, threshold: float | None = None,
    ) -> SanitizationResult:
        """Scan a single piece of content the agent did NOT receive
        directly from the user — a web page, a RAG chunk, a tool output.
        This is the indirect-injection surface. For scanning several
        chunks at once, prefer pre_process_external_batch, which actually
        parallelizes across CPU cores."""
        result = self.sanitizer.scan_text(
            content,
            threshold=threshold if threshold is not None else self.input_risk_threshold,
            tag="EXTERNAL_CONTENT",
            source_id=source_id,
        )
        await self.audit.log("external_content_scan", {
            "source_id": source_id,
            "risk_score": result.risk_score,
            "matched_patterns": result.matched_patterns,
            "matched_languages": list(result.matched_languages),
            "decoded_payload_hits": bool(result.decoded_payload_hits),
            "blocked": result.blocked,
        })
        return result

    async def pre_process_external_batch(
        self, chunks: list[tuple[str, str]], threshold: float | None = None,
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
            self.audit.log("external_content_scan", {
                "source_id": result.source_id,
                "risk_score": result.risk_score,
                "matched_patterns": result.matched_patterns,
                "matched_languages": list(result.matched_languages),
                "decoded_payload_hits": bool(result.decoded_payload_hits),
                "blocked": result.blocked,
            })
            for result in results
        ])

        return list(results)

    # -- Output ---------------------------------------------------------

    async def post_process(self, model_output: str) -> PostProcessResult:
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
            )
        else:
            result, exfil = _scan_output(
                self.output_guard,
                self.exfil_guard if self.scan_output_for_exfil else None,
                model_output,
                self.system_prompt,
                self.output_overlap_threshold,
            )
        blocked = result.blocked or (exfil is not None and exfil.blocked)

        await self.audit.log("output_scan", {
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
        await self.audit.log("ingest", {
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
            # The guard sent to the worker processes is a copy without the
            # provenance store: that store owns a connection, which is not
            # picklable and has no business being duplicated per worker.
            # Recording stays in this process, where the connection lives.
            scanner = dataclasses.replace(self.ingest_guard, provenance_store=None)
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
            self.audit.log("ingest", {
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
            await self.audit.log("retrieval_provenance", {
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
        result = await self.media_scanner.scan(payload, media_type=media_type, source_id=source_id)
        if session_id is not None and result.risk_score > 0:
            await self.rate_limiter.record_turn_risk(
                self.session_key(session_id, principal), result.risk_score, actor_id=actor_id,
            )
        await self.audit.log("media_scan", {
            "session_id": session_id,
            "principal": principal,
            "source_id": source_id,
            "media_type": media_type,
            "extractors": [e.extractor for e in result.extractions],
            "extractor_errors": result.extractor_errors,
            "risk_score": result.risk_score,
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
        await self.rate_limiter.check_tool_call(self.session_key(session_id, principal))

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
        """
        self._require_principal(principal, "authorized_tool_call")
        if session_id is not None:
            await self.check_tool_call_budget(session_id, principal)
        if self.scan_tool_call_arguments:
            # Checked before the scope check runs, because an in-scope
            # action with an attacker-chosen destination is the whole
            # problem: `http_get` may be perfectly authorized.
            findings = self.exfil_guard.scan_values([args, kwargs])
            if findings and self.enforcement == "shadow":
                await self.audit.log("tool_call", {
                    "session_id": session_id, "principal": principal,
                    "action": action, "status": "would_deny",
                    "reason": "exfiltration_in_arguments",
                    "exfil_reasons": sorted({r for f in findings for r in f.reasons}),
                })
            elif findings:
                await self.audit.log("tool_call", {
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
            await self.audit.log("tool_call", {
                "session_id": session_id, "principal": principal,
                "action": action, "status": "allowed",
            })
            return output
        except Exception as exc:
            await self.audit.log("tool_call", {
                "session_id": session_id, "principal": principal,
                "action": action, "status": "denied", "reason": str(exc),
            })
            raise
