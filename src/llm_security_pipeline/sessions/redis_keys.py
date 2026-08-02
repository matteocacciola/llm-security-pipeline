"""
redis_keys.py
Cluster-safe key construction for the Redis-backed stores.

Why this module exists: Redis Cluster shards the keyspace into 16384 hash
slots, and a Lua script may only touch keys that live in the same slot.
`add_risk` touches three keys together (risk, risk_last, flagged) and
`reset_session` deletes five; with plain `prefix:<session_id>:<field>`
naming those land in different slots and the server rejects the call with
CROSSSLOT before running a single line of the script.

The fix is a hash tag: Redis hashes only the substring between the first
`{` and the first `}` after it, so `prefix:{<session_id>}:risk` and
`prefix:{<session_id>}:flagged` are guaranteed to share a slot while
different sessions still spread evenly across the cluster.

Two sharp edges this module handles, both of which silently reintroduce
CROSSSLOT if ignored:

* An empty tag is not a tag. `{}` makes Redis fall back to hashing the
  whole key, so a session id of "" would break the invariant. Session ids
  come from the calling application and are not validated anywhere else,
  so they cannot be assumed brace-free or non-empty.
* A brace already present in `key_prefix` wins, because Redis uses the
  FIRST brace group in the key. Every session would then collapse into a
  single slot. The stores reject such a prefix at construction rather
  than quietly creating a hot shard.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

# Marks a tag that had to be derived rather than used verbatim. A raw
# session id shaped exactly like "#<16 hex chars>" would alias with a
# derived tag; the consequence is two sessions sharing rate/risk counters,
# not a privilege boundary crossing, and the shape is unlikely enough in
# practice to be worth keeping keys human-readable in redis-cli for the
# overwhelmingly common case.
_DERIVED_TAG_MARKER = "#"


def hash_tag(session_id: str) -> str:
    """Return the hash-tag content to use for `session_id`.

    Verbatim for ordinary ids, so operators can still find a session's keys
    by eye. Ids that would produce a broken or empty tag (empty string, or
    anything containing a brace) fall back to a digest, which is stable
    across processes and restarts because it is a pure function of the id.
    """
    if not session_id or "{" in session_id or "}" in session_id:
        digest = hashlib.blake2s(session_id.encode("utf-8"), digest_size=8).hexdigest()
        return f"{_DERIVED_TAG_MARKER}{digest}"
    return session_id


def validate_key_prefix(key_prefix: str) -> None:
    """Reject a prefix containing braces.

    Earlier versions of this library documented `key_prefix` as the place
    to put a hash tag manually. That is now handled automatically, and a
    brace left in the prefix would take precedence over the generated tag
    (Redis reads the first brace group only), pinning every session in the
    deployment to one slot.
    """
    if "{" in key_prefix or "}" in key_prefix:
        raise ValueError(
            f"key_prefix must not contain braces (got {key_prefix!r}). Hash tags "
            "for Redis Cluster are applied automatically per session id; a brace "
            "in the prefix would take precedence and pin every session to a "
            "single hash slot. Drop the braces from the prefix."
        )


def _is_cluster_client(client: object) -> bool:
    """True if `client` is a redis-py cluster client, without any I/O."""
    try:
        from redis.asyncio.cluster import RedisCluster
    except ImportError:  # pragma: no cover - redis missing or too old
        return False
    return isinstance(client, RedisCluster)


class HashTagPolicy:
    """Decides whether session keys get wrapped in a hash tag.

    Modes:
        "auto"  (default) - detect the topology on first use and cache it.
        True              - always tag. Valid on a single node too; braces
                            are ordinary characters there.
        False             - never tag. Preserves a pre-existing keyspace on
                            single-node deployments; raises on first use if
                            the client is a cluster client, so the mistake
                            surfaces at startup instead of as a CROSSSLOT
                            error on some later request.

    Detection is deliberately lazy: store constructors are synchronous and
    an `INFO` round trip cannot be issued from one. It is also idempotent
    and unsynchronised — several coroutines racing on the very first call
    may each issue one `INFO`, which is cheaper than carrying a lock that
    would bind this object to a single event loop.
    """

    __slots__ = ("_mode", "_resolved", "_checked_explicit_false")

    def __init__(self, mode: bool | str = "auto"):
        if mode not in (True, False, "auto"):
            raise ValueError(f"hash_tags must be True, False or 'auto' (got {mode!r})")
        self._mode = mode
        self._resolved: bool | None = None if mode == "auto" else bool(mode)
        self._checked_explicit_false = False

    @property
    def mode(self) -> bool | str:
        return self._mode

    @property
    def resolved(self) -> bool | None:
        """The decision, or None while still undetermined in auto mode."""
        return self._resolved

    async def enabled(self, client) -> bool:
        if self._mode is False:
            if not self._checked_explicit_false:
                self._checked_explicit_false = True
                if _is_cluster_client(client):
                    raise RuntimeError(
                        "hash_tags=False was requested with a Redis Cluster client. "
                        "Multi-key operations (add_risk, reset_session) would fail "
                        "with CROSSSLOT. Use hash_tags=True or the 'auto' default."
                    )
            return False
        if self._resolved is None:
            self._resolved = await self._detect(client)
        return self._resolved

    async def _detect(self, client) -> bool:
        if _is_cluster_client(client):
            return True
        try:
            info = await client.info("cluster")
            enabled = bool(int(info.get("cluster_enabled", 0) or 0))
        except Exception as exc:  # managed Redis may restrict INFO
            # Tagged keys are correct on both topologies, so an
            # inconclusive probe resolves towards the safe answer.
            logger.warning(
                "Could not determine Redis topology (%s); assuming Redis Cluster "
                "and applying hash tags to session keys. Set hash_tags=False "
                "explicitly if this is a single node and you need the previous "
                "key layout.",
                exc,
            )
            return True
        if enabled:
            logger.info("Redis Cluster detected; applying hash tags to session keys.")
        return enabled
