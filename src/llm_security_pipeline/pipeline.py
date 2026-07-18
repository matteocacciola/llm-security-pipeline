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
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .services import (
    Sanitizer,
    SanitizationResult,
    ScopeGuard,
    OutputGuard,
    OutputScanResult,
    SessionRateLimiter,
    SessionLimits,
)
from .state_backend import StateBackend, RedisStateBackend

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover
    Redis = None  # type: ignore


@dataclass
class PreProcessResult:
    sanitized: SanitizationResult
    blocked: bool


@dataclass
class PostProcessResult:
    scan: OutputScanResult
    safe_text: str
    blocked: bool


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

        if scope_guard is None:
            scope_guard = ScopeGuard(
                nonce_store=state_backend.nonce_store if state_backend is not None else None
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

    # -- Direct input (the user's own message) -----------------------------

    async def pre_process(self, user_input: str, session_id: str | None = None) -> PreProcessResult:
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
        if session_id is not None:
            result, _ = await asyncio.gather(scan_awaitable, self.rate_limiter.check_request(session_id))
        else:
            result = await scan_awaitable

        cumulative = None
        session_flagged = False
        if session_id is not None:
            cumulative = await self.rate_limiter.record_turn_risk(session_id, result.risk_score)
            session_flagged = await self.rate_limiter.is_session_flagged(session_id)

        await self.audit.log("input_scan", {
            "session_id": session_id,
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
        return PreProcessResult(sanitized=result, blocked=result.blocked or session_flagged)

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

        if len(chunks) < self.external_scan_parallel_min_chunks:
            results = [
                self.sanitizer.scan_text(content, threshold=effective_threshold, tag="EXTERNAL_CONTENT", source_id=source_id)
                for content, source_id in chunks
            ]
        else:
            futures = [
                loop.run_in_executor(
                    self.process_executor,
                    self.sanitizer.scan_text,
                    content,
                    effective_threshold,
                    "EXTERNAL_CONTENT",
                    source_id,
                )
                for content, source_id in chunks
            ]
            results = await asyncio.gather(*futures)

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
        # OutputGuard.scan is pure CPU (regex); offload only if the text is
        # large enough for that to matter (same rationale as pre_process).
        if len(model_output) >= self.large_input_offload_threshold_chars:
            result = await loop.run_in_executor(
                self.process_executor, self.output_guard.scan, model_output,
                self.system_prompt, self.output_overlap_threshold,
            )
        else:
            result = self.output_guard.scan(
                model_output, system_prompt=self.system_prompt, overlap_threshold=self.output_overlap_threshold,
            )

        await self.audit.log("output_scan", {
            "secret_categories": list(result.secret_findings.keys()),
            "pii_categories": list(result.pii_findings.keys()),
            "system_prompt_overlap_score": result.system_prompt_overlap_score,
            "blocked": result.blocked,
        })
        safe_text = result.redacted_text if not result.blocked else (
            "[Response blocked by the security layer: possible credential "
            "or system-prompt leak detected.]"
        )
        return PostProcessResult(scan=result, safe_text=safe_text, blocked=result.blocked)

    # -- Tool calling / budget -------------------------------------------

    async def check_tool_call_budget(self, session_id: str) -> None:
        """Raises RateLimitExceeded if the session has made too many tool
        calls within the current window — mitigates DoS/cost abuse via
        excessive tool usage, independent of whether each individual call
        was in-scope."""
        await self.rate_limiter.check_tool_call(session_id)

    async def authorized_tool_call(
        self, token, action: str, func: Callable, *args, session_id: str | None = None, **kwargs,
    ):
        if session_id is not None:
            await self.check_tool_call_budget(session_id)
        try:
            output = await self.scope_guard.guarded_call(token, action, func, *args, **kwargs)
            await self.audit.log("tool_call", {"session_id": session_id, "action": action, "status": "allowed"})
            return output
        except Exception as exc:
            await self.audit.log("tool_call", {
                "session_id": session_id, "action": action, "status": "denied", "reason": str(exc),
            })
            raise
