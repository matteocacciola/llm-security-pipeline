# Installation and packaging

Installing with uv or pip, the src layout, the optional extras, and whether you need Redis at all.

[← Back to README](../README.md)

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

The in-memory stores are correct for one process and evict what has
expired on an opportunistic sweep, so a single-process deployment does not
grow without bound; flags expire with the session, as they do on the
shared backends.


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
