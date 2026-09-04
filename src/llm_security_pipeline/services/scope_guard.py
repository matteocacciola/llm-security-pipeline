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
import logging
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..resilience import (
    Degraded,
    TOKEN_REPLAY,
    FailurePolicy,
    ResilientBackend,
)
from ..sessions.stores import NonceStore, InMemoryNonceStore

logger = logging.getLogger(__name__)


class ScopeError(Exception):
    pass


class UnknownKeyId(ScopeError):
    """The token names a signing key this guard does not hold.

    Distinct from an invalid signature because the two mean different
    things operationally: a bad signature is a forgery or a key mismatch,
    while an unknown key id is almost always a rotation that was retired
    too early or rolled out in the wrong order.
    """


# ---------------------------------------------------------------------------
# Signing keys
# ---------------------------------------------------------------------------
# A capability token is signed once and verified later, possibly in another
# process, and it stays valid for its TTL. That makes the signing key a
# thing you have to be able to change without a window in which valid
# tokens are rejected — and with a single key there is no such change: the
# instant any process starts signing with a new key, every token already in
# flight becomes "possible tampering" everywhere else.
#
# So the key is named. `kid` travels inside the signed payload, verification
# looks up that name, and a guard can hold several keys while signing with
# exactly one. Rotation is then three deploys and no downtime; see
# `SigningKeyring` and the README.

# Key ids end up in JSON payloads, error messages and audit logs, all of
# which are read by something. Restricting the alphabet keeps an
# attacker-supplied `kid` from carrying newlines into a log line.
_KID_RE = re.compile(r"\A[A-Za-z0-9._:-]{1,64}\Z")

# HMAC-SHA256 gains nothing from a key longer than its block size and loses
# real security below its output size. 32 bytes is the floor, and it is
# enforced rather than documented because "we used a short key" is not a
# mistake that announces itself.
MIN_KEY_BYTES = 32

DEFAULT_KEY_ID = "default"


_AUDIENCE_RE = re.compile(r"\A[A-Za-z0-9._:/-]{1,128}\Z")


def _check_audience(audience: str) -> str:
    if not isinstance(audience, str) or not _AUDIENCE_RE.match(audience):
        raise ValueError(
            f"audience must be 1-128 characters of [A-Za-z0-9._:/-], got {audience!r}."
        )
    return audience


def _check_kid(kid: str, where: str) -> str:
    if not isinstance(kid, str) or not _KID_RE.match(kid):
        raise ValueError(
            f"{where}: key id must be 1-64 characters of [A-Za-z0-9._:-], "
            f"got {kid!r}."
        )
    return kid


@dataclass(frozen=True)
class SigningKeyring:
    """The keys a ScopeGuard signs and verifies with.

    One key is `active` and does the signing; every key in `keys` is
    accepted at verification. That asymmetry is the whole feature: it is
    what lets a new key be everywhere before anything signs with it.

        keyring = SigningKeyring.single(current_key, kid="2026-01")

        # 1. everyone accepts the new key, nobody signs with it yet
        keyring = keyring.with_key("2026-02", new_key)

        # 2. only once step 1 is fully rolled out, promote it
        keyring = keyring.with_active("2026-02")

        # 3. once the longest token TTL has elapsed, drop the old one
        keyring = keyring.without_key("2026-01")

    The ordering is not a style preference. Promote before every process
    accepts the new key and tokens signed by an updated process arrive at
    one that has never heard of that key id, which fails exactly like the
    single-key case this exists to avoid — except now the error names the
    key, so you can see it.
    """

    keys: Mapping[str, bytes]
    active: str

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("SigningKeyring needs at least one key.")
        # Copy into a plain dict so a caller's mutable mapping can't change
        # under a guard that has already been constructed.
        frozen = dict(self.keys)
        for kid, key in frozen.items():
            _check_kid(kid, "SigningKeyring")
            if not isinstance(key, (bytes, bytearray)):
                raise TypeError(f"SigningKeyring: key {kid!r} must be bytes, got {type(key).__name__}.")
            if len(key) < MIN_KEY_BYTES:
                raise ValueError(
                    f"SigningKeyring: key {kid!r} is {len(key)} bytes; "
                    f"HMAC-SHA256 signing keys must be at least {MIN_KEY_BYTES}. "
                    "Generate one with secrets.token_bytes(32)."
                )
        _check_kid(self.active, "SigningKeyring")
        if self.active not in frozen:
            raise ValueError(
                f"SigningKeyring: active key id {self.active!r} is not in keys "
                f"({', '.join(sorted(frozen))})."
            )
        object.__setattr__(self, "keys", MappingProxyType({k: bytes(v) for k, v in frozen.items()}))

    # The keys are held in a MappingProxyType so a caller cannot mutate a
    # keyring a guard is already using. That proxy is not picklable, which
    # would otherwise surface as a confusing failure the first time
    # anything carrying a ScopeGuard crossed a process boundary.
    def __getstate__(self) -> dict[str, object]:
        return {"keys": dict(self.keys), "active": self.active}

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "keys", state["keys"])
        object.__setattr__(self, "active", state["active"])
        self.__post_init__()

    @classmethod
    def single(cls, key: bytes, kid: str = DEFAULT_KEY_ID) -> "SigningKeyring":
        """One key, which is the right shape until the first rotation."""
        return cls(keys={kid: key}, active=kid)

    @classmethod
    def generated(cls) -> "SigningKeyring":
        """A random key, private to this process. Only correct for a single
        process; ScopeGuard warns when it falls back to this."""
        return cls.single(secrets.token_bytes(MIN_KEY_BYTES), kid="ephemeral")

    def signing_key(self) -> tuple[str, bytes]:
        return self.active, self.keys[self.active]

    def verification_key(self, kid: str) -> bytes:
        """Look up an accepted key by id.

        A direct lookup, never a try-every-key fallback: the id arrives
        inside an unverified payload, so the only safe thing to do with it
        is index a dictionary and fail on a miss.
        """
        try:
            return self.keys[kid]
        except (KeyError, TypeError):
            raise UnknownKeyId(
                f"Token was signed with key id {kid!r}, which this guard does not hold. "
                f"Accepted ids: {', '.join(sorted(self.keys))}. Either the key was "
                "retired while tokens signed with it were still valid, or a process "
                "was promoted to sign with a new key before every process accepted it."
            ) from None

    def with_key(self, kid: str, key: bytes) -> "SigningKeyring":
        """Accept an additional key without signing with it yet. Step 1."""
        if kid in self.keys:
            raise ValueError(f"SigningKeyring already holds key id {kid!r}.")
        return SigningKeyring(keys={**self.keys, kid: key}, active=self.active)

    def with_active(self, kid: str) -> "SigningKeyring":
        """Start signing with a key that is already accepted. Step 2."""
        if kid not in self.keys:
            raise ValueError(
                f"Cannot sign with key id {kid!r}: add it with with_key() and roll "
                "that out everywhere first, or tokens it signs will be rejected by "
                "any process that has not caught up."
            )
        return SigningKeyring(keys=dict(self.keys), active=kid)

    def without_key(self, kid: str) -> "SigningKeyring":
        """Stop accepting a key. Step 3, and only after the longest token
        TTL has elapsed since it stopped signing."""
        if kid == self.active:
            raise ValueError(
                f"Cannot retire key id {kid!r} while it is the active signing key."
            )
        if kid not in self.keys:
            raise ValueError(f"SigningKeyring does not hold key id {kid!r}.")
        return SigningKeyring(
            keys={k: v for k, v in self.keys.items() if k != kid}, active=self.active,
        )

    @property
    def key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.keys))


# The exact bytes the signature covers. Signing a re-serialization of the
# payload is a trap: `issued_at`/`expires_at` are floats, and a token that
# crosses a process boundary as JSON is re-parsed into floats that need not
# render identically, at which point a perfectly valid token fails with
# "possible tampering". So the canonical bytes are produced once and, when a
# token is parsed from the wire, kept verbatim rather than recomputed.
def canonical_payload_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True).encode()


# Bound on what `from_str` will parse before it has verified anything.
# Nothing legitimate is anywhere near this, and the parse happens on
# attacker-supplied input.
MAX_TOKEN_CHARS = 16_384

_REQUIRED_PAYLOAD_FIELDS = ("agent_id", "scopes", "expires_at", "nonce", "max_uses", "kid")


@dataclass
class CapabilityToken:
    payload: dict
    signature: str
    # Set when the token was parsed from its serialized form; the bytes the
    # issuer actually signed. None for a freshly issued token, where the
    # payload dict is authoritative.
    raw: bytes | None = None

    def canonical(self) -> bytes:
        return self.raw if self.raw is not None else canonical_payload_bytes(self.payload)

    @property
    def key_id(self) -> str | None:
        """Which signing key this token names. Inside the signed payload,
        so it cannot be repointed at another key without invalidating the
        signature — and repointing it would not help anyway, since the
        attacker would still have to produce a signature under that key."""
        return self.payload.get("kid")

    @property
    def audience(self) -> str | None:
        return self.payload.get("aud")

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
        """Serialize the token so it can cross a process or network boundary.

        The counterpart of `from_str`. Note that this is not confidential:
        the payload is encoded, not encrypted, and anyone holding the string
        can read the scopes and subject. What the signature guarantees is
        that they cannot change them.
        """
        return self.canonical().hex() + "." + self.signature

    @classmethod
    def from_str(cls, token_str: str) -> "CapabilityToken":
        """Parse a token produced by `to_str`.

        This does NOT verify anything — it cannot, since the key lives in
        the ScopeGuard. Pass the result to `authorize`, which checks the
        signature before it looks at any field. Malformed input raises
        ScopeError rather than ValueError/JSONDecodeError, so a caller
        handling untrusted input has one exception type to catch.
        """
        if not isinstance(token_str, str):
            raise ScopeError("Token must be a string.")
        if len(token_str) > MAX_TOKEN_CHARS:
            raise ScopeError(f"Token exceeds {MAX_TOKEN_CHARS} characters; refusing to parse.")

        payload_hex, separator, signature = token_str.partition(".")
        if not separator or not payload_hex or not signature:
            raise ScopeError("Malformed token: expected '<payload_hex>.<signature>'.")

        try:
            raw = bytes.fromhex(payload_hex)
        except ValueError as exc:
            raise ScopeError("Malformed token: payload is not valid hex.") from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScopeError("Malformed token: payload is not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ScopeError("Malformed token: payload is not an object.")

        # The signature is verified over `raw`, so this parse cannot be used
        # to smuggle fields in — but failing here gives a clear error
        # instead of a KeyError deep inside authorize().
        missing = [f for f in _REQUIRED_PAYLOAD_FIELDS if f not in payload]
        if missing:
            raise ScopeError(f"Malformed token: payload is missing {', '.join(missing)}.")

        return cls(payload=payload, signature=signature, raw=raw)


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
        keyring: SigningKeyring | None = None,
        failure_policy: FailurePolicy | None = None,
        resilience: ResilientBackend | None = None,
        audience: str | None = None,
    ):
        # Which service this guard IS. Tokens it issues name it, and tokens
        # it verifies must name it back. The README tells you to give every
        # process the same signing key — and it has to — but the moment
        # two different services share that key, each accepts the other's
        # tokens unless something in the token says which service it was
        # for. A subject binds a token to a user; an audience binds it to
        # a service. Optional for the single-service case, which is where
        # nearly everyone starts; set it the day a second service appears.
        self._audience = _check_audience(audience) if audience is not None else None
        if secret_key is not None and keyring is not None:
            raise ValueError(
                "Pass either secret_key or keyring, not both. secret_key is shorthand "
                "for SigningKeyring.single(secret_key); build the keyring yourself as "
                "soon as you need to rotate."
            )

        # A generated key is private to THIS process. That is fine for a
        # single process and silently wrong the moment there are two: a
        # token issued on one pod fails verification on every other one,
        # because they never agreed on a key. It is the same class of bug as
        # the per-process nonce dict this library exists to fix, except it
        # surfaces as "possible tampering" on a token that was never
        # tampered with — so it is worth saying out loud at construction
        # rather than leaving to be debugged from a misleading error.
        self._ephemeral_key = secret_key is None and keyring is None
        if self._ephemeral_key:
            logger.warning(
                "ScopeGuard: no secret_key or keyring was supplied, so a random key was "
                "generated for this process. Capability tokens issued here cannot be "
                "verified by any other process or pod. Supply the same key everywhere "
                "(from your secret manager) for any deployment running more than one "
                "process; a generated key is only correct for a single-process app, "
                "tests or a prototype."
            )
            keyring = SigningKeyring.generated()
        elif keyring is None:
            keyring = SigningKeyring.single(secret_key)  # type: ignore[arg-type]
        self._keyring = keyring

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
        # Defaults to fail-closed, and that default is not a coin toss:
        # without the nonce store there is no max_uses, so a token that
        # should be spendable once becomes spendable without limit.
        # Refusing a tool call is bounded and recoverable; an unbounded
        # capability token is not.
        self._resilience = resilience or ResilientBackend(failure_policy)

    @property
    def audience(self) -> str | None:
        return self._audience

    @property
    def keyring(self) -> SigningKeyring:
        return self._keyring

    @property
    def active_key_id(self) -> str:
        """Which key this guard is currently signing with. Worth exporting
        during a rotation: it is how you confirm a deploy actually took."""
        return self._keyring.active

    def _sign_bytes(self, raw: bytes, key: bytes) -> str:
        return hmac.new(key, raw, hashlib.sha256).hexdigest()

    def _sign(self, payload: dict) -> str:
        return self._sign_bytes(canonical_payload_bytes(payload), self._keyring.signing_key()[1])

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
        kid, key = self._keyring.signing_key()
        payload: dict = {
            "agent_id": agent_id,
            "scopes": sorted(set(scopes)),
            "issued_at": time.time(),
            "expires_at": time.time() + ttl_seconds,
            "nonce": secrets.token_hex(16),
            "max_uses": max_uses,
            "subject": subject,
            "constraints": dict(constraints or {}),
            # Signed, so it cannot be repointed at another key — not that
            # repointing would help, since producing a signature under that
            # key is the part an attacker cannot do.
            "kid": kid,
            # Also signed. None when this guard has no audience, which is
            # the single-service case; a verifying guard with an audience
            # rejects a None here, so a token minted "for nobody" cannot
            # be spent at a service that expects to be named.
            "aud": self._audience,
        }
        signature = self._sign_bytes(canonical_payload_bytes(payload), key)
        return CapabilityToken(payload=payload, signature=signature)

    def _verify_signature(self, token: CapabilityToken) -> None:
        # The payload must never be mutated after issuance: this signature
        # is computed once, at issue_token time, over the exact payload
        # dict. Any later in-place edit (e.g. writing a "uses" counter into
        # it) would make this check fail for a legitimate token. That's why
        # usage tracking lives entirely in the external nonce store instead.
        # Which key to check against is named by the token. The name is
        # untrusted at this point, so it is used to index a dictionary and
        # nothing else: no trying every key in turn, which would turn each
        # verification into a search over the whole keyring and make a
        # retired key indistinguishable from a current one.
        kid = token.payload.get("kid")
        if not isinstance(kid, str):
            raise ScopeError(
                "Token names no signing key (missing 'kid'). It was issued by a "
                "differently-configured authority, or by a version of this library "
                "from before key rotation existed."
            )
        try:
            key = self._keyring.verification_key(kid)
        except UnknownKeyId as exc:
            if self._ephemeral_key:
                raise UnknownKeyId(
                    "This ScopeGuard is using a randomly generated per-process key, so "
                    "it holds no key the rest of your deployment knows about — this is "
                    "what a token issued by any other process looks like. Give every "
                    "process the same explicit secret_key or keyring."
                ) from exc
            raise

        expected = self._sign_bytes(token.canonical(), key)
        if not hmac.compare_digest(expected, str(token.signature)):
            if self._ephemeral_key:
                raise ScopeError(
                    "Invalid token signature. This ScopeGuard is using a randomly "
                    "generated per-process key, so this is what a token issued by a "
                    "different process looks like — check that every process shares "
                    "one explicit secret_key before treating it as tampering."
                )
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

        # Audience before subject: "wrong service" is the more fundamental
        # mismatch, and its error should not be masked by a subject check
        # that happens to also fail.
        if self._audience is not None:
            presented = token.payload.get("aud")
            if presented is None:
                raise ScopeError(
                    f"Token names no audience but this guard is {self._audience!r}. "
                    "It was issued by a guard without an audience — a token for "
                    "nobody in particular is not a token for this service."
                )
            if not hmac.compare_digest(str(presented), self._audience):
                raise ScopeError(
                    f"Token was issued for {presented!r}, not for {self._audience!r}. "
                    "The two services share a signing key; the audience is what "
                    "keeps their tokens from being interchangeable."
                )

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
        uses = await self._resilience.run(
            TOKEN_REPLAY,
            lambda: self._nonce_store.check_and_increment(
                token.payload["nonce"], token.payload["max_uses"], remaining_ttl
            ),
        )
        if isinstance(uses, Degraded):
            # Fail-open on replay: the token is honoured without a use
            # count. Recorded as a backend_degraded event, because this is
            # the one window in which max_uses means nothing.
            return
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
