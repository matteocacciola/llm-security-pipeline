# llm_security_pipeline

A defensive pipeline for LLM-based applications: direct and indirect prompt
injection, unauthorized tool calling, output/PII/secret leaks, and
session-level abuse (DoS, multi-turn jailbreak build-up). This version is
async and Redis-backed, built for multi-process/multi-pod cloud deployments.

## Structure

- `services/sanitizer.py` – Unicode normalization, encoded-payload
  detection, explicit data-wrapping, multilingual lexical heuristics
  (IT/EN/ES/FR/DE/PT). Pure CPU, no I/O — safe to run in a worker process.
  `Sanitizer` holds a compiled `PatternRegistry` (same
  `pattern_config_path`/`include_default_patterns`/`registry` constructor
  shape as `OutputGuard` below); `scan_text` is a convenience wrapper over
  a lazily-created default instance.
- `services/scope_guard.py` – signed capability tokens (HMAC-SHA256) with
  scope, TTL and anti-replay for gating agent tool calls. Async; delegates
  replay tracking to a pluggable `NonceStore`.
- `services/output_guard.py` – credential/PII detection and system-prompt
  leak detection. Patterns come from `config_loader.py` /
  `config/patterns.json`.
- `services/rate_limiter.py` – per-session request/tool-call budget and
  cumulative cross-turn risk tracking. Async; delegates counters to a
  pluggable `SessionStore`.
- `config_loader.py` – loads and validates secret/PII/injection-phrase
  regex patterns from external JSON, extensible without touching library
  code.
- `sessions/stores.py` – backend-agnostic interfaces (`NonceStore`,
  `SessionStore`) for the pipeline's shared mutable state, plus in-memory
  implementations for local development, unit tests and single-process
  deployments.
- `sessions/redis_stores.py` – Redis implementations of those interfaces,
  using atomic Lua scripts so state is correct under concurrent access
  from multiple processes. Requires the optional `redis` extra.
- `sessions/postgres_stores.py` – PostgreSQL implementations, using atomic
  single-statement upserts (`INSERT ... ON CONFLICT ... RETURNING`) with
  logical TTL columns and opportunistic cleanup. Requires the optional
  `postgres` extra (asyncpg).
- `sessions/mysql_stores.py` – MySQL/MariaDB implementations: single-statement
  `LAST_INSERT_ID()` upserts for integer counters, and a locked
  transaction with bounded deadlock retry (InnoDB error 1213) for the
  float risk accumulator. Requires the optional `mysql` extra (aiomysql).
- `state_backend.py` – `StateBackend`, the technology-neutral bundle of
  stores the pipeline consumes, with three ready-made implementations:
  `RedisStateBackend`, `PostgresStateBackend` and `MySQLStateBackend`.
  Implement the two interfaces in `sessions/stores.py` on anything else
  (e.g. DynamoDB) to use a different technology with no changes elsewhere.
- `pipeline.py` – async orchestrator (`SecurityPipeline`), with structured
  audit logging (stdout JSON by default, Redis Streams optional) and a
  `ProcessPoolExecutor` for the batch-scanning workload that actually
  benefits from multi-core parallelism.

## Why this needed a rewrite for production

The original single-process version kept its rate-limit counters and
token-replay tracking in plain Python dicts. That is only safe if your
application runs as exactly one process forever. In any real cloud
deployment — multiple gunicorn/uvicorn workers, multiple pods behind a load
balancer, autoscaling — different requests land on different processes,
each with its own private copy of those dicts. Concretely, this means:

- A capability token with `max_uses=1` could be redeemed once **per
  process** instead of once total — an attacker (or a buggy retry) hitting
  N different pods gets N uses out of a "single-use" token.
- A session's request budget or cumulative jailbreak-risk score would only
  see the fraction of traffic that happened to land on one process,
  silently multiplying the effective limit by the number of instances.

This was verified directly, not just reasoned about:
`tests/test_token_replay.py` spawns 5 separate OS processes that all race
to redeem the same single-use token, against each of the three backends in
turn. With the old in-memory store, all 5 succeed (the bug). With
`RedisNonceStore` (from `sessions/redis_stores.py`), exactly 1 succeeds and
the other 4 are correctly denied as replay attempts — atomicity is
enforced by Redis executing each check-and-increment as a single
uninterruptible Lua script, not by Python-level locking (which can't help
across separate processes anyway). The PostgreSQL and MySQL backends give
the same guarantee via their own atomic primitives (see
`sessions/postgres_stores.py` / `sessions/mysql_stores.py`).

## Installation and packaging (uv, src layout)

This is a **library**, consumed by your application — it has no Dockerfile
or deployment of its own. It uses `pyproject.toml` with uv (no
`requirements.txt`) and the `src` layout:

```
project-root/
├── pyproject.toml
├── uv.lock
└── src/
    └── llm_security_pipeline/
        ├── __init__.py
        ├── ... (all modules)
        └── config/
            └── patterns.json    # must live INSIDE the package: config_loader.py
                                 # resolves it relative to its own __file__
```

The package directory must be `src/llm_security_pipeline/`, not loose
modules in `src/`: with hatchling, the last path component of `packages`
becomes the installed package name, so loose files in `src/` would install
a package literally named `src` (name collisions guaranteed the moment two
libraries do the same). Common commands:

```bash
uv sync                  # create .venv, install from lockfile (dev)
uv run pytest            # run tests in the managed environment
uv build                 # produce the wheel + sdist in dist/
unzip -l dist/*.whl | grep patterns.json   # verify the config file made it in
```

All shared-state backends are **optional extras**, not hard dependencies:

```bash
uv add llm-security-pipeline              # base: in-memory stores only
uv add "llm-security-pipeline[redis]"     # Redis backend
uv add "llm-security-pipeline[postgres]"  # PostgreSQL backend (asyncpg)
uv add "llm-security-pipeline[mysql]"     # MySQL/MariaDB backend (aiomysql)
```

Note that `uv.lock` pins versions for *this repo's* development and CI
only; consumers of the library resolve against the version ranges declared
in `pyproject.toml`, as is correct for a library.

## Do you actually need Redis?

Not always. The decision depends on **how many Python processes share the
limits being enforced**, not on how many machines you have:

- **No Redis needed** — local development and tests (the in-memory stores
  are the default), single-process deployments (a CLI tool, a bot with one
  worker, a prototype), or any usage that only touches the stateless parts
  of the pipeline (input sanitization, external-content scanning, output
  guard) without capability tokens or session rate limiting.
- **Redis required** — anything with more than one process serving the
  same traffic: `uvicorn/gunicorn --workers N` with N > 1 (yes, even on a
  single VM — that is already N independent processes), multiple pods
  behind a load balancer, autoscaling, serverless. The failure mode
  without it is silent and demonstrated in `tests/test_token_replay.py`
  and `tests/test_rate_limit.py`: each process keeps private counters, so
  a `max_uses=1` token redeems once *per process* and session budgets
  multiply by the instance count, with no error anywhere.
- **Alternatives to Redis** — `PostgresStateBackend` and `MySQLStateBackend`
  are ready-made if you already run one of those; for anything else, the
  `NonceStore`/`SessionStore` interfaces in `sessions/stores.py` are
  abstract precisely so you can back them with DynamoDB (conditional
  writes) or similar, at the cost of somewhat higher per-operation
  latency; bundle your implementations in a `StateBackend` and pass it to
  `SecurityPipeline(state_backend=...)`. Sticky sessions at the load balancer are **not** a substitute:
  they are fragile under pod churn and do nothing against token replay,
  which can arrive outside the session's process.

## Quick start (production)

```python
from llm_security_pipeline import SecurityPipeline, RedisStateBackend

async def handle_request(user_message: str, session_id: str, retrieved_chunks: list[str]):
    async with SecurityPipeline(
        system_prompt=SYSTEM_PROMPT,
        state_backend=RedisStateBackend.from_url("redis://redis-service:6379/0"),
        pii_config_path="config/my_org_patterns.json",
    ) as pipeline:

        pre = await pipeline.pre_process(user_message, session_id=session_id)
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

## Testing

Tests live in `tests/` and run with pytest (`asyncio_mode = "auto"`, see
`pyproject.toml`):

- `test_in_memory_stores.py` – plain unit tests for
  `InMemoryNonceStore`/`InMemorySessionStore`: no infrastructure, no
  `ProcessPoolExecutor`, just the single-process logic they're actually
  meant for (counter increments, window reset, risk decay/flagging,
  session reset).
- `test_config_loader.py` – plain unit tests for `PatternRegistry`: default
  loading, override-by-name and add-alongside semantics, and every
  `PatternConfigError` path (malformed JSON, missing file, missing
  fields, unknown flag, invalid regex, zero patterns).
- `test_sanitizer.py` – plain unit tests for `scan_text`/`Sanitizer`:
  multilingual phrase matching (by config-defined name, not raw regex
  text), the code-mixing score bonus, encoded-payload detection, the
  block threshold, and `Sanitizer`'s override knobs (custom
  `pattern_config_path`, `include_default_patterns=False`, an explicit
  `registry`).
- `test_token_replay.py` – 5 real OS processes race to redeem the same
  `max_uses=1` capability token; asserts exactly 1 succeeds. Parametrized
  over all three backends (`redis`, `postgres`, `mysql`) — each is
  supposed to provide the identical guarantee on different technology.
- `test_rate_limit.py` – 6 real OS processes each send 1 request against a
  shared session budget of 3; asserts exactly 3 are allowed. Same
  three-backend parametrization as the token-replay test.
- `test_parallel_scanning.py` – asserts a batch scanned via
  `ProcessPoolExecutor` matches a sequential scan of the same chunks,
  including a deliberately injected malicious one. Redis-only: this test
  is about CPU parallelism, not about the state backend.
- `test_end_to_end.py` – exercises `SecurityPipeline` front-to-back: a
  benign message, indirect injection in external content, and a leaked
  secret in model output. Redis-only, for the same reason.

The cross-process tests are marked `@pytest.mark.integration` plus one of
`@pytest.mark.redis` / `postgres` / `mysql`: they need a reachable
instance of that backend (real cross-process state, not an in-memory
stand-in) and spawn real OS processes, so they don't fit the "unit test"
label despite using `assert`. `tests/conftest.py` checks reachability for
each backend once at collection time and skips only the parametrized
cases for backends that aren't reachable — so, for example, running with
only Redis up still exercises the Redis cases and cleanly skips the
PostgreSQL/MySQL ones instead of failing.

```bash
uv sync --extra redis --extra postgres --extra mysql   # drivers used by the test suite

docker run -d --rm -p 6379:6379 redis:7-alpine
docker run -d --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-alpine
docker run -d --rm -e MYSQL_ALLOW_EMPTY_PASSWORD=yes -e MYSQL_DATABASE=test -p 3306:3306 mysql:8

uv run pytest
```

Override `REDIS_URL` / `POSTGRES_DSN` / `MYSQL_HOST` (+ `MYSQL_PORT`,
`MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`) if your instances aren't on
the defaults above. `uv run pytest` is always safe to run even with no
backend available at all — every integration case skips instead of
failing, and the unit tests still run and report real coverage on the
in-memory stores.

## Deployment notes

- **Redis**: any managed Redis (ElastiCache, Memorystore, Azure Cache,
  Redis Cloud) works; no special modules required beyond stock Redis
  (`EVAL`/Lua scripting is core functionality). Use a dedicated logical DB
  or key prefix per environment to avoid staging/prod key collisions —
  `RedisNonceStore`/`RedisSessionStore` accept a `key_prefix` argument.
- **Connection reuse**: pass one shared `redis.asyncio.Redis` client (which
  manages its own connection pool) to `RedisStateBackend(redis_client)`
  rather than letting every request open a new connection.
- **Process pool sizing**: defaults to `os.cpu_count()` workers. Size it to
  your container's actual CPU allocation/limit, not the host's — on
  Kubernetes with CPU limits, `os.cpu_count()` reports the *node's* core
  count, not your pod's quota, and can over-provision workers. Pass
  `process_executor=ProcessPoolExecutor(max_workers=N)` explicitly if you
  set CPU limits.
- **Graceful shutdown**: always call `pipeline.aclose()` (or use
  `async with`) so the process pool's worker processes and the Redis
  connection are released cleanly on deploy/restart.
- **Redis Streams audit log**: `RedisStreamAuditLogger` is capped with
  `maxlen` (approximate trimming) so it doesn't grow unbounded; for durable
  long-term storage, consume the stream into your log/SIEM pipeline rather
  than relying on Redis itself as the archive.

## Where the parallelism actually is (and where it deliberately isn't)

This is the part most likely to be over-engineered if done carelessly, so
here is the reasoning, not just the result:

| Operation | Bound by | Strategy | Why |
|---|---|---|---|
| Redis reads/writes (budget check, risk update, token replay check) | I/O | `asyncio` + `asyncio.gather` | Network round-trips are naturally concurrent; no GIL contention. |
| Single direct user message scan | CPU (regex) | Inline, synchronous | For typical message sizes this takes microseconds; a process-pool round trip (serialize args, queue, deserialize result) costs more than the work itself. Offloading it would make the common case *slower*. |
| Batch of external/RAG content chunks | CPU (regex), several independent items | `ProcessPoolExecutor` via `run_in_executor`, gathered | This is the one place where the CPU work is large enough, and split into genuinely independent units, that spreading it across real OS processes (bypassing the GIL) pays for itself. Below `external_scan_parallel_min_chunks` (default 2) it still runs inline. |
| Very large pasted input (> `large_input_offload_threshold_chars`, default 20,000 chars) | CPU (regex) | Offloaded to the process pool | Large enough that the regex work dominates the IPC overhead. |
| High request throughput in general | — | Run multiple OS processes of *your application* (uvicorn/gunicorn workers, multiple pods) | This is the standard way Python achieves multi-core throughput for many concurrent requests, and it's exactly why the Redis-backed shared state exists: so those independent processes agree on one set of limits instead of each enforcing its own. |

A single-core sandbox can't demonstrate a speedup, so
`tests/test_parallel_scanning.py` verifies the parallel batch path for
**correctness** (its output matches sequential execution) rather than for
speed. On a real multi-core host, the same code parallelizes across cores
without any changes.

## Extending patterns via config.json

Secret, PII **and** multilingual injection-phrase patterns all live in
`config/patterns.json` and share the same override mechanism, via
`config_loader.py`'s `PatternRegistry`: an entry in a custom file with the
same `name` as a bundled default REPLACES it, a new `name` is added
alongside the rest. `injection_patterns` entries carry one extra required
field, `lang`, since that heuristic groups by language (used for the
code-mixing signal in `scan_text`) — otherwise they're the same
`{name, pattern, flags}` shape as `secret_patterns`/`pii_patterns`.

- `OutputGuard`/`SecurityPipeline(pii_config_path=..., include_default_pii_patterns=...)`
  picks up custom secret/PII patterns.
- `Sanitizer`/`SecurityPipeline(injection_config_path=..., include_default_injection_patterns=...)`
  picks up custom injection phrases, the same way. `Sanitizer` mirrors
  `OutputGuard`'s constructor exactly: `pattern_config_path`,
  `include_default_patterns`, or a pre-built `registry` for full control.
  The module-level `scan_text` function is a thin convenience wrapper
  around a lazily-created default `Sanitizer()` for callers who don't need
  any of that.

Every entry is validated against a pydantic schema before it's compiled:
required fields (`name`/`pattern`, plus `lang` for injection entries),
unknown fields rejected (`extra="forbid"`), unknown flag names, and
regex syntax — all reported as a `PatternConfigError` pointing at the
exact entry (e.g. `secret_patterns.2.pattern`) rather than surfacing as a
confusing error somewhere in the scanning path, or a pattern silently not
applying at all. `pydantic` is consequently a **required** dependency of
this library (not an extra like `redis`/`postgres`/`mysql`): pattern
loading always runs, for every install, regardless of which shared-state
backend you pick.

See the module docstring in `config_loader.py` for the exact schema.

## Threat coverage

**Covered:**
- Direct and indirect prompt injection
- Unauthorized tool calling (scope-gated capability tokens)
- Secret/PII/system-prompt leakage in model output
- DoS/cost-abuse via excessive requests or tool calls (session budget)
- Multi-turn jailbreak build-up (cumulative session risk)
- **Correctness of all of the above under multi-process/multi-pod
  deployment** (this revision's focus)

**Still NOT covered:**
- Data poisoning of a knowledge base/RAG index at ingestion time
- Exfiltration via secondary channels (e.g. rendered markdown image URLs)
- Non-text jailbreak vectors (e.g. hidden text in images)
- End-user authentication/authorization upstream of the agent
- Any guarantee against a sufficiently novel/creative attack — heuristics
  and thresholds need tuning and red-teaming against your specific agent

## Important design choices (read before integrating)

1. **The primary defense is structural, not lexical.** A single lexical
   injection-phrase match, by design, does not cross the default block
   threshold. The real protection is `wrap_as_data()` plus a system prompt
   that states content in `USER_DATA`/`EXTERNAL_CONTENT` tags is never an
   instruction, in any language.
2. **Capability tokens must never be mutated after issuance** — the
   signature covers the exact payload set at issue time. Usage tracking
   lives entirely outside the token, in the nonce store.
3. **In-memory stores (`InMemoryNonceStore`, `InMemorySessionStore`) are
   for local development and tests only.** They silently stop providing
   any real guarantee the moment you run more than one process — there is
   no error or warning at runtime, because from a single process's point
   of view everything looks correct. Use the Redis-backed stores (or wire
   a `StateBackend` into `SecurityPipeline`) for any real
   deployment.
4. **Phone-number PII detection is intentionally permissive** and produces
   false positives; treat it as a signal to review, not an automatic block.

## Honest limitations

- Lexical pattern lists cover 6 common languages; extend based on your
  real user base.
- The Lua scripts assume a single Redis node/primary (standard for
  ElastiCache/Memorystore-style managed Redis). A Redis Cluster deployment
  needs the keys touched together by a script (`risk`, `risk_last`,
  `flagged`) to hash to the same slot — the default key naming does not
  guarantee this; use Redis Cluster hash tags (`{session_id}`) in
  `key_prefix` if you deploy on a cluster.
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
