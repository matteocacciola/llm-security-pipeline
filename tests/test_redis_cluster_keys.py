"""
Redis Cluster correctness for the session store.

Two layers here, and the split is deliberate. The slot-invariant and
detection tests use redis-py's own CRC16 slot function and fake clients,
so they run in ordinary CI with nothing listening: they pin the property
that actually matters (every key a script touches for one session lands in
one slot) rather than the string format that happens to produce it. The
tests marked `redis_cluster` then confirm the same thing against a real
cluster-enabled server, which is the only place a genuine CROSSSLOT error
can be observed; they skip automatically when none is reachable.
"""

from __future__ import annotations

import pytest
from redis.crc import key_slot

from llm_security_pipeline.sessions.redis_keys import (
    HashTagPolicy,
    hash_tag,
    validate_key_prefix,
)
from llm_security_pipeline.sessions.redis_stores import _SESSION_FIELDS, RedisSessionStore

from conftest import REDIS_CLUSTER_URL

# Session ids a caller might realistically hand us, plus the ones that
# break naive tagging: empty (a `{}` tag is not a tag at all, so Redis
# falls back to hashing the whole key) and brace-carrying ones.
SESSION_IDS = [
    "user-123-session-456",
    "s",
    "",
    "{}",
    "a}b{c",
    "}leading",
    "trailing{",
    "sessão-übung-会话",
    "x" * 512,
]


class _FakeClient:
    """Stands in for a standalone client; records INFO calls."""

    def __init__(self, cluster_enabled: int = 0, raises: Exception | None = None):
        self._cluster_enabled = cluster_enabled
        self._raises = raises
        self.info_calls = 0

    def register_script(self, script: str):
        return object()

    async def info(self, section: str | None = None):
        self.info_calls += 1
        if self._raises is not None:
            raise self._raises
        return {"cluster_enabled": self._cluster_enabled}


def _store(client, **kwargs) -> RedisSessionStore:
    return RedisSessionStore(client, **kwargs)


# ---------------------------------------------------------------------------
# Key layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("session_id", SESSION_IDS)
async def test_all_session_keys_share_one_slot(session_id):
    store = _store(_FakeClient(cluster_enabled=1))
    keys = await store._keys(session_id, *_SESSION_FIELDS)
    slots = {key_slot(key.encode()) for key in keys}
    assert len(slots) == 1, f"{session_id!r} spread over slots {slots}"


@pytest.mark.parametrize("session_id", SESSION_IDS)
async def test_add_risk_key_group_shares_one_slot(session_id):
    """The three keys the risk script touches together, specifically."""
    store = _store(_FakeClient(cluster_enabled=1))
    keys = await store._keys(session_id, "risk", "risk_last", "flagged")
    assert len({key_slot(key.encode()) for key in keys}) == 1


async def test_distinct_sessions_still_spread_across_slots():
    """Tagging must shard by session, not funnel everything into one slot."""
    store = _store(_FakeClient(cluster_enabled=1))
    slots = set()
    for i in range(200):
        (key,) = await store._keys(f"session-{i}", "risk")
        slots.add(key_slot(key.encode()))
    assert len(slots) > 100


async def test_untagged_layout_is_the_documented_failure_case():
    """Guards the reason this module exists: without tags the risk script's
    keys land in different slots, which a cluster rejects outright."""
    store = _store(_FakeClient(cluster_enabled=0))
    keys = await store._keys("user-123", "risk", "risk_last", "flagged")
    assert len({key_slot(key.encode()) for key in keys}) > 1


async def test_tagged_keys_keep_the_session_id_readable():
    store = _store(_FakeClient(cluster_enabled=1))
    (key,) = await store._keys("user-123", "risk")
    assert key == "sentinel:session:{user-123}:risk"


async def test_untagged_layout_is_unchanged_from_previous_versions():
    store = _store(_FakeClient(cluster_enabled=0))
    (key,) = await store._keys("user-123", "risk")
    assert key == "sentinel:session:user-123:risk"


def test_hash_tag_is_verbatim_when_safe():
    assert hash_tag("user-123") == "user-123"


@pytest.mark.parametrize("session_id", ["", "{}", "a}b{c", "}x"])
def test_hash_tag_falls_back_to_a_digest_when_unsafe(session_id):
    tag = hash_tag(session_id)
    assert "{" not in tag and "}" not in tag and tag != ""
    assert tag == hash_tag(session_id)  # stable across calls/processes


def test_hash_tag_fallback_keeps_sessions_distinct():
    assert hash_tag("{a}") != hash_tag("{b}")


# ---------------------------------------------------------------------------
# key_prefix validation
# ---------------------------------------------------------------------------

def test_braced_prefix_is_rejected():
    """The old manual workaround, now a hot-slot bug: a brace in the prefix
    is the first brace group in the key, so it beats the per-session tag."""
    with pytest.raises(ValueError, match="must not contain braces"):
        validate_key_prefix("sentinel:{session}:")


def test_store_rejects_braced_prefix_at_construction():
    with pytest.raises(ValueError, match="must not contain braces"):
        _store(_FakeClient(), key_prefix="sentinel:{app}:session:")


# ---------------------------------------------------------------------------
# Topology detection
# ---------------------------------------------------------------------------

async def test_auto_detects_cluster_and_caches_the_probe():
    client = _FakeClient(cluster_enabled=1)
    store = _store(client)
    for _ in range(5):
        await store._keys("s", "risk")
    assert client.info_calls == 1


async def test_auto_detects_single_node():
    client = _FakeClient(cluster_enabled=0)
    policy = HashTagPolicy("auto")
    assert await policy.enabled(client) is False


async def test_unreachable_info_resolves_to_tagging():
    """Some managed Redis offerings restrict INFO. Tagged keys are valid on
    both topologies, so an inconclusive probe resolves that way."""
    client = _FakeClient(raises=RuntimeError("unknown command 'INFO'"))
    policy = HashTagPolicy("auto")
    assert await policy.enabled(client) is True


async def test_explicit_modes_skip_the_probe():
    for mode in (True, False):
        client = _FakeClient(cluster_enabled=1)
        policy = HashTagPolicy(mode)
        assert await policy.enabled(client) is mode
        assert client.info_calls == 0


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="hash_tags"):
        HashTagPolicy("yes")


async def test_opting_out_on_a_cluster_client_fails_fast():
    """Better a startup error than a CROSSSLOT surfacing on some later
    request, in the middle of the guard path."""
    pytest.importorskip("redis.asyncio.cluster")
    from redis.asyncio.cluster import RedisCluster

    client = RedisCluster.__new__(RedisCluster)  # no connection attempted
    policy = HashTagPolicy(False)
    with pytest.raises(RuntimeError, match="CROSSSLOT"):
        await policy.enabled(client)


async def test_cluster_client_needs_no_probe():
    pytest.importorskip("redis.asyncio.cluster")
    from redis.asyncio.cluster import RedisCluster

    client = RedisCluster.__new__(RedisCluster)
    assert await HashTagPolicy("auto").enabled(client) is True


# ---------------------------------------------------------------------------
# Against a real cluster-enabled server
# ---------------------------------------------------------------------------

@pytest.fixture
async def cluster_store():
    from redis.asyncio.cluster import RedisCluster

    client = RedisCluster.from_url(REDIS_CLUSTER_URL)
    store = RedisSessionStore(client, key_prefix="test:cluster:session:")
    yield store
    await client.aclose()


@pytest.mark.integration
@pytest.mark.redis_cluster
async def test_add_risk_runs_on_a_real_cluster(cluster_store):
    session_id = "cluster-session-1"
    await cluster_store.reset_session(session_id)
    first = await cluster_store.add_risk(session_id, 0.5, 0.0, 1.0, 60)
    second = await cluster_store.add_risk(session_id, 0.6, 0.0, 1.0, 60)
    assert first == pytest.approx(0.5)
    assert second == pytest.approx(1.1)
    assert await cluster_store.is_flagged(session_id) is True
    await cluster_store.reset_session(session_id)
    assert await cluster_store.is_flagged(session_id) is False


@pytest.mark.integration
@pytest.mark.redis_cluster
@pytest.mark.parametrize("session_id", ["", "{}", "a}b{c", "sessão-会话"])
async def test_awkward_session_ids_run_on_a_real_cluster(cluster_store, session_id):
    """The digest fallback has to hold up where it counts: the server, not
    a local CRC calculation."""
    await cluster_store.reset_session(session_id)
    assert await cluster_store.add_risk(session_id, 0.3, 0.0, 99.0, 60) == pytest.approx(0.3)
    assert await cluster_store.increment_requests(session_id, 60) == 1
    await cluster_store.reset_session(session_id)


@pytest.mark.integration
@pytest.mark.redis_cluster
async def test_untagged_keys_would_be_rejected_by_the_cluster(cluster_store):
    """Proves the tags are load-bearing rather than decorative: the same
    script with the old key layout is refused by the server."""
    from redis.exceptions import RedisError

    keys = [f"test:cluster:session:untagged:{f}" for f in ("risk", "risk_last", "flagged")]
    with pytest.raises((RedisError, Exception), match="(?i)slot"):
        await cluster_store._add_risk_script(keys=keys, args=[0.1, 0.0, 0.0, 60, 1.0])
