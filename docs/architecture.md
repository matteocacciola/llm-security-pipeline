# Architecture and design

How the library is put together, what is deliberately parallel and what deliberately is not, and the design choices that shape the API. Read this before integrating.

[← Back to README](../README.md)

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
4. **External-content risk is not charged to a session unless you ask.**
   `pre_process` and `pre_process_media` feed their risk score into the
   session's cumulative total; `pre_process_external`/`_batch` do so only
   when given a `session_id`. Both readings are defensible and only you
   know which applies — a poisoned page the user never chose is not
   evidence about the user, while a user steering retrieval at a document
   they planted is. When enabled, the worst chunk in a batch is charged
   rather than the sum, so a wide retrieval does not flag a session for
   being wide.
5. **Phone-number PII detection is intentionally permissive** and produces
   false positives; treat it as a signal to review, not an automatic block.
