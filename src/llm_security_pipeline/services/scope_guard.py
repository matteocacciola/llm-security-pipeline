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

    @property
    def subject(self) -> str | None:
        """The authenticated end user this token was issued for, if any.

        Inside the signed payload, so it cannot be swapped for someone
        else's without invalidating the token.
        """
        return self.payload.get("subject")

    @property
    def constraints(self) -> dict:
        """Signed, caller-defined limits carried alongside the scope (for
        example `{"account_id": "42"}`).

        The library transports and protects these; it cannot enforce them,
        because whether account 42 belongs to this user is a question about
        your data model. See `check_constraints`.
        """
        return self.payload.get("constraints") or {}

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

    def __init__(
        self,
        secret_key: bytes | None = None,
        nonce_store: NonceStore | None = None,
        require_subject: bool = False,
    ):
        self._secret_key = secret_key or secrets.token_bytes(32)
        # When True, a token must name the end user it was issued for, and
        # that user must be presented again at authorization. This is what
        # stops a leaked token from being spendable by whoever finds it —
        # the confused-deputy case, where the agent acts with the token's
        # authority regardless of who is asking now.
        self._require_subject = require_subject
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
        subject: str | None = None,
        constraints: dict | None = None,
    ) -> CapabilityToken:
        """Issue a token. `subject` binds it to an authenticated end user;
        `constraints` carries signed, caller-defined limits.

        The library never authenticates `subject` — it receives it. What it
        guarantees is that the value cannot be altered after issuance and
        must match at authorization.
        """
        if self._require_subject and subject is None:
            raise ScopeError(
                "This ScopeGuard requires tokens to be bound to a subject "
                "(require_subject=True), but issue_token was called without one. "
                "Pass the authenticated end-user identifier as subject=."
            )
        payload = {
            "agent_id": agent_id,
            "scopes": sorted(set(scopes)),
            "issued_at": time.time(),
            "expires_at": time.time() + ttl_seconds,
            "nonce": secrets.token_hex(16),
            "max_uses": max_uses,
            "subject": subject,
            "constraints": dict(constraints or {}),
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

    async def authorize(
        self, token: CapabilityToken, action: str, subject: str | None = None,
    ) -> None:
        """Verify that the token is valid, not expired, not reused beyond
        its allowed limit, and that the requested action is within the
        granted scope. Raises ScopeError if any condition is not met.

        The replay check is the only part that needs to be atomic across
        processes, so it's the only part delegated to the nonce store.
        """
        self._verify_signature(token)

        bound_subject = token.payload.get("subject")
        if bound_subject is not None:
            if subject is None:
                raise ScopeError(
                    "Token is bound to a subject but no subject was presented at "
                    "authorization. Pass the authenticated end user as subject=."
                )
            if not hmac.compare_digest(str(bound_subject), str(subject)):
                raise ScopeError(
                    "Token was issued for a different end user than the one "
                    "presenting it (possible token theft or replay across users)."
                )
        elif self._require_subject:
            raise ScopeError(
                "This ScopeGuard requires subject-bound tokens, but this token "
                "carries no subject. It was issued by a differently-configured "
                "authority, or before binding was enabled."
            )

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

    @staticmethod
    def check_constraints(token: CapabilityToken, **actual) -> None:
        """Compare the token's signed constraints against the values a call
        is actually about, raising if any differ.

        This is the mechanism, not the policy. It answers "does this call
        match what the token was issued for", which the library can verify
        cryptographically. It does not answer "should this user be allowed
        near account 42", which depends on your data model and stays in
        your tool.
        """
        for key, expected in token.constraints.items():
            if key not in actual:
                raise ScopeError(f"Token constrains '{key}' but the call did not supply it.")
            if str(actual[key]) != str(expected):
                raise ScopeError(
                    f"Token was issued for {key}={expected!r}, call is for {actual[key]!r}."
                )

    async def guarded_call(
        self, token: CapabilityToken, action: str, func, *args, subject: str | None = None, **kwargs,
    ):
        """Convenience wrapper: executes func(*args, **kwargs) only if authorized.
        `func` may be sync or async; async functions are awaited."""
        await self.authorize(token, action, subject=subject)
        result = func(*args, **kwargs)
        if hasattr(result, "__await__"):
            result = await result
        return result
