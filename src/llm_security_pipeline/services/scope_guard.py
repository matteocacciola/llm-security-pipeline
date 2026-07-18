"""
scope_guard.py
Layer 2 of the pipeline: enforcement of permissions on the actions
("tool calls") an LLM agent may perform. This is independent of the
language the request arrived in — here we gate the ACTION, not the text.

Core idea (conceptually inspired by the "capability token" pattern used by
several agent security gateways): the model never decides on its own
whether an action is allowed. Every tool call requires a signed token, with
an explicit scope, a TTL, and anti-replay protection, issued by an
authority separate from the model itself. If the requested action is not in
the token's scope, it is blocked at the infrastructure level — not left to
the model's "judgment".

Production note: token usage/replay tracking is delegated to a NonceStore
(see stores.py / redis_stores.py). Use RedisNonceStore when running multiple
processes/pods so a token's max_uses limit can't be bypassed by replaying
it against a different process than the one that saw it first.
"""

from __future__ import annotations

import hmac
import hashlib
import json
import secrets
import time
from dataclasses import dataclass

from ..sessions.stores import NonceStore, InMemoryNonceStore


class ScopeError(Exception):
    pass


@dataclass
class CapabilityToken:
    payload: dict
    signature: str

    def to_str(self) -> str:
        raw = json.dumps(self.payload, sort_keys=True).encode()
        return raw.hex() + "." + self.signature


class ScopeGuard:
    """Authority that issues and verifies capability tokens for tool calling.

    Typical usage:
        guard = ScopeGuard(secret_key=..., nonce_store=RedisNonceStore(redis_client))
        token = guard.issue_token(agent_id="sales_bot", scopes=["read_crm"], ttl_seconds=60)
        await guard.authorize(token, action="read_crm")   # ok
        await guard.authorize(token, action="send_email")  # raises ScopeError

    `issue_token` and signature verification are pure/synchronous (no I/O);
    only `authorize`, which needs to consult shared usage state, is async.
    """

    def __init__(self, secret_key: bytes | None = None, nonce_store: NonceStore | None = None):
        self._secret_key = secret_key or secrets.token_bytes(32)
        # Defaults to an in-memory store, which is only safe for a single
        # process. Pass a RedisNonceStore explicitly for any multi-process
        # or multi-instance deployment.
        self._nonce_store = nonce_store or InMemoryNonceStore()

    def _sign(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True).encode()
        return hmac.new(self._secret_key, raw, hashlib.sha256).hexdigest()

    def issue_token(
        self,
        agent_id: str,
        scopes: list[str],
        ttl_seconds: int = 300,
        max_uses: int = 1,
    ) -> CapabilityToken:
        payload = {
            "agent_id": agent_id,
            "scopes": sorted(set(scopes)),
            "issued_at": time.time(),
            "expires_at": time.time() + ttl_seconds,
            "nonce": secrets.token_hex(16),
            "max_uses": max_uses,
        }
        signature = self._sign(payload)
        return CapabilityToken(payload=payload, signature=signature)

    def _verify_signature(self, token: CapabilityToken) -> None:
        # The payload must never be mutated after issuance: this signature
        # is computed once, at issue_token time, over the exact payload
        # dict. Any later in-place edit (e.g. writing a "uses" counter into
        # it) would make this check fail for a legitimate token. That's why
        # usage tracking lives entirely in the external nonce store instead.
        expected = self._sign(token.payload)
        if not hmac.compare_digest(expected, token.signature):
            raise ScopeError("Invalid token signature: possible tampering.")

    async def authorize(self, token: CapabilityToken, action: str) -> None:
        """Verify that the token is valid, not expired, not reused beyond
        its allowed limit, and that the requested action is within the
        granted scope. Raises ScopeError if any condition is not met.

        The replay check is the only part that needs to be atomic across
        processes, so it's the only part delegated to the nonce store.
        """
        self._verify_signature(token)

        now = time.time()
        if now > token.payload["expires_at"]:
            raise ScopeError("Token expired.")

        if action not in token.payload["scopes"]:
            raise ScopeError(
                f"Action '{action}' not authorized for agent "
                f"'{token.payload['agent_id']}'. Granted scope: {token.payload['scopes']}"
            )

        remaining_ttl = max(1, int(token.payload["expires_at"] - now))
        uses = await self._nonce_store.check_and_increment(
            token.payload["nonce"], token.payload["max_uses"], remaining_ttl
        )
        if uses > token.payload["max_uses"]:
            raise ScopeError("Token already used the maximum allowed number of times (possible replay).")

    async def guarded_call(self, token: CapabilityToken, action: str, func, *args, **kwargs):
        """Convenience wrapper: executes func(*args, **kwargs) only if authorized.
        `func` may be sync or async; async functions are awaited."""
        await self.authorize(token, action)
        result = func(*args, **kwargs)
        if hasattr(result, "__await__"):
            result = await result
        return result
