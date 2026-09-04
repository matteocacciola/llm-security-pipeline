# Deployment

Running in production: multi-process correctness, the signing key, connection reuse, configuration as data, what happens when the backend is unreachable, and input size limits.

[← Back to README](../README.md)

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
  fix is to pass the key from your secret manager. Once you need to change
  that key without downtime, pass `scope_keyring=` instead; see
  [Rotating the signing key](identity-and-tokens.md#rotating-the-signing-key).
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

## Configuration as data

`SecurityPipeline.__init__` takes some thirty keyword arguments, and they
are two kinds of thing: *settings* (a threshold, a mode, a path) that could
live in a YAML file and are identical in every process, and *objects* (a
Redis client, a process pool, a sink) that are built per process and cannot
be written down. `PipelineConfig` separates them:

```python
config = PipelineConfig.from_dict(yaml.safe_load(open("guard.yaml")))
pipeline = SecurityPipeline.from_config(
    config,
    state_backend=backend,      # objects still come from code
    audit_logger=RedisStreamAuditLogger(redis),
    metrics=PrometheusMetricsSink(),
)
```

`from_dict` refuses unknown keys rather than ignoring them — a misspelled
setting that silently falls back to its default is a security posture that
is not the one written down — and every value is validated at construction.
Passing a setting beside a config is refused too, since with two sources
for one value nothing in the call says which won. The signing key is
deliberately *not* a setting, so a config file that gets committed does not
become a key that gets committed. `pipeline.config_summary` gives the
running posture back as data, for a startup log line or a health endpoint.

## When the backend is down

Every degraded decision is also counted in `degradations_total`; see
[Metrics](detection-and-tuning.md#metrics).


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

## Input size

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
