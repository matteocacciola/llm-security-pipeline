# llm_security_pipeline

A defensive pipeline for LLM-based applications: direct and indirect prompt
injection, unauthorized tool calling, output/PII/secret leaks, and
session-level abuse (DoS, multi-turn jailbreak build-up). This version is
async and Redis-backed, built for multi-process/multi-pod cloud deployments.

```
pip install "llm-security-pipeline[redis]"
```

Other extras: `postgres`, `mysql`, `images`, `metrics`, or `all`. Full
details, the `uv` workflow and the src layout are in
[docs/installation.md](docs/installation.md).

## Quick start (production)

```python
from llm_security_pipeline import SecurityPipeline, RedisStateBackend

async def handle_request(user_message: str, session_id: str, user_id: str, retrieved_chunks: list[str]):
    async with SecurityPipeline(
        system_prompt=SYSTEM_PROMPT,
        session_identity="authenticated",   # your gateway verified session_id
        scope_secret_key=SIGNING_KEY,        # same value in every process
        state_backend=RedisStateBackend.from_url("redis://redis-service:6379/0"),
        pii_config_path="config/my_org_patterns.json",
    ) as pipeline:

        pre = await pipeline.pre_process(user_message, session_id=session_id, principal=user_id)
        if pre.blocked:
            return "Request blocked for security reasons."

        chunks = [(text, f"rag:{i}") for i, text in enumerate(retrieved_chunks)]
        external = await pipeline.pre_process_external_batch(chunks)  # parallelized across CPU cores

        model_output = await call_llm(build_prompt(pre.sanitized, external))

        post = await pipeline.post_process(model_output)
        return post.safe_text
```

In a long-lived service, construct `SecurityPipeline` once at startup
(reusing one backend/connection pool and one process pool across requests)
rather than per-request; `async with` in the example above is for
demonstration. Call `await pipeline.aclose()` on shutdown.

A complete, running service — YAML config, SSE streaming with the
replace-on-block contract, a tool behind a scoped token, `/metrics`,
`/health` — is in [`examples/fastapi_chat/`](examples/fastapi_chat/app.py);
its tests run in CI so it cannot rot.

That is the whole integration surface for the common case. The rest of
this page says what it covers and where to read more; the details live in
`docs/`.

## Threat coverage

**Covered:**
- Direct and indirect prompt injection
- Unauthorized tool calling (scope-gated capability tokens)
- Secret/PII/system-prompt leakage in model output
- DoS/cost-abuse via excessive requests or tool calls (session budget)
- Multi-turn jailbreak build-up (cumulative session risk)
- Exfiltration through rendered side channels — markdown/HTML image and
  media URLs, CSS `url()`, and URLs passed as tool-call arguments
- Instructions smuggled in invisible codepoints (Unicode Tags block,
  directional overrides)
- **Correctness of all of the above under multi-process/multi-pod
  deployment**

- Knowledge-base/RAG poisoning at ingestion, with provenance verification
  at retrieval
- Non-text carriers, through pluggable extractors (metadata and embedded
  strings out of the box; OCR is yours to supply)

- Rotating the token signing key without a window in which valid tokens
  are rejected
- Revoking every token a subject holds, and delegating with tokens that
  can only shrink
- Rotating the signing key without a restart, reviewing quarantined
  documents, budgets per tier, and a health endpoint that knows which
  breakers are open
- Homoglyph attacks (`ignоre` with a Cyrillic о) and another tenant's
  identifiers appearing in a reply
- Measuring all of the above — shadow-mode deltas, score distributions,
  detector calibration — as aggregate metrics with no identifiers in them
- Plugging in a model-based detector to cover the paraphrase gap the
  lexical heuristics leave open, without shipping a model
- Scanning a response while it streams, without emitting anything that has
  not been seen whole
- Deciding, per guard, whether an unreachable state backend means "refuse"
  or "proceed unchecked" — and making the second one visible when it
  happens
- Binding tokens and session state to an authenticated end user, when your
  gateway supplies one, plus risk accumulation that survives session
  rotation when it does not

**Still NOT covered:**
- **Authenticating anyone.** The library receives a principal, it cannot
  verify one. That belongs at your gateway and no future version will
  change it — see [Identity](docs/identity-and-tokens.md#identity) for exactly where the boundary falls.
- **Object-level authorization.** Whether account 42 belongs to this user
  is a question about your data model. The library transports and verifies
  signed constraints; the policy stays in your tool.
- Text rendered as pixels, unless you plug in OCR. See [Non-text input](docs/guards.md#non-text-input-mediascanner).
- Any guarantee against a sufficiently novel attack. No code produces one.
  What the library now provides instead is the means to measure: see
  [Tuning the thresholds](docs/detection-and-tuning.md#tuning-the-thresholds).

## Documentation

Read in this order the first time; dip in afterwards.

| Page | What it is for |
| --- | --- |
| [Architecture and design](docs/architecture.md) | How it is put together, what is parallel and what deliberately is not, and the design choices that shape the API. **Read before integrating.** |
| [Installation](docs/installation.md) | uv/pip, extras, src layout, and whether you need Redis at all. |
| [Deployment](docs/deployment.md) | Multi-process correctness, the signing key, connection reuse, configuration as data (`PipelineConfig`), what happens when the backend is down, input size limits. |
| [Identity and capability tokens](docs/identity-and-tokens.md) | Where the identity boundary falls, binding tokens and session state to a principal, rotating the signing key without downtime. |
| [The guards](docs/guards.md) | Ingestion-time scanning, non-text input, the canary, streaming output, side-channel exfiltration, invisible-codepoint smuggling. |
| [Detection, tuning and measurement](docs/detection-and-tuning.md) | Extending the patterns, semantic detectors, choosing thresholds, shadow mode, metrics. |
| [Testing](docs/testing.md) | What each test file checks and why, the backend integration tests, the lint and type gates, the audit schema version. |

Also: [CHANGELOG.md](CHANGELOG.md) for what changed and why, and
[SECURITY.md](SECURITY.md) for how to report a vulnerability and what is
in scope.

## Three things to know before you deploy

The full list is in [Architecture and design](docs/architecture.md#important-design-choices-read-before-integrating);
these are the ones people get wrong.

1. **Declare `session_identity`.** Say whether the `session_id` you pass
   was authenticated by your gateway (`"authenticated"`, with a
   `principal=` on each call) or not (`"untrusted"`, with an `actor_id=` so
   risk still lands somewhere stable). Left undeclared, the pipeline falls
   back to `"untrusted"` and warns on every construction — it will not
   guess "authenticated", because that binds session state to a claim
   nobody verified.
2. **Pass the same signing key to every process.** Left unset, each
   process generates its own, and a token issued on one pod fails on every
   other one. Both the guard and the pipeline warn when this happens.
3. **The thresholds are product decisions, not facts.** The defaults are
   reasonable in general and therefore wrong for your traffic in
   particular. Run in shadow mode, look at the metrics, then enforce.

## Honest limitations

- Lexical pattern lists cover 6 common languages; extend based on your
  real user base.
- Redis Cluster is supported (see [Deployment notes](docs/deployment.md#deployment-notes)), but a session's
  counters live in one slot by construction, so a single very hot session
  is served by a single shard. This is inherent to keeping the risk update
  atomic; sessions spread evenly across the cluster, individual sessions do
  not spread at all.
- The SQL backends emulate TTL with `expires_at` columns: expiry is
  enforced logically in every query (correctness never depends on
  cleanup), but physical row deletion is opportunistic — if you need
  strict storage bounds, schedule your own periodic cleanup job.
- The MySQL backend retries on InnoDB deadlock (error 1213) with bounded
  backoff — this is the documented InnoDB usage pattern, verified under
  real cross-process contention, but it means p99 latency under heavy
  same-key contention is higher than Redis or PostgreSQL.
- No heuristic substitutes for actual red-teaming of your specific agent.
- This code has not been through an external security audit.
