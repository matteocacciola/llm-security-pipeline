# Identity and capability tokens

Where the identity boundary falls, binding tokens and session state to a principal, and rotating the signing key without downtime.

[← Back to README](../README.md) · Deployment-side notes on the key are in [Deployment](deployment.md#deployment-notes).

## Identity

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

## Audience: which service a token is for

The deployment notes say to give every process the same signing key, and
that is right — but the moment two *different services* share that key,
each accepts the other's tokens. A token minted for the orders service is
spendable at the payments service if the scope name happens to match. A
subject binds a token to a user; an audience binds it to a service.

```python
orders   = ScopeGuard(secret_key=KEY, audience="orders")
payments = ScopeGuard(secret_key=KEY, audience="payments")

token = orders.issue_token("bot", ["read"])
await payments.authorize(token, "read")   # ScopeError: issued for 'orders', not for 'payments'
```

`aud` is inside the signed payload, so it cannot be rewritten. A guard with
an audience rejects a token that names none — a token "for nobody" is not a
token for this service — while a guard *without* an audience accepts any
token, so a single-service deployment needs no change. It is checked before
the subject, so "wrong service" is not masked by a subject mismatch that
happens to fail too. The pipeline takes it as `scope_audience=`, and
`PipelineConfig` as `audience` (it is an identity, not a secret, so it
belongs in config; the key does not).

## Revoking a subject

Tokens are bearer credentials with a TTL, and until now the only answer to
"that user was phished ten minutes ago" was to wait the TTL out.
`pipeline.revoke_subject(subject)` records the instant in the nonce store;
every token of that subject with `issued_at` at or before it is refused,
tokens issued afterwards are not, so the subject can be re-issued without
un-revoking anything. The check runs before the nonce is spent, so a
refused token does not consume a use. The revocation record lives
`ttl_seconds` (default a day) and **must outlive the longest token TTL you
issue**. Revoking is fail-closed on the write: a caller told "revoked" must
not find the tokens still working. Audited as `subject_revoked`.

## Delegating with an attenuated token

An orchestrator holding a token for five actions asks a sub-agent to do
one. Passing the token down gives it all five. `pipeline.attenuate(parent,
scopes=[...], ttl_seconds=, max_uses=, constraints=)` derives a token that
can only shrink — a subset of the scopes, an expiry no later than the
parent's, at most as many uses, every parent constraint kept and only new
ones added, the same subject and audience — with each rule enforced by the
guard, not trusted from the caller. The child carries `parent` (the
parent's nonce) and `depth` in its signed payload, so the chain reads back
from the audit log; depth is capped at `MAX_DELEGATION_DEPTH` (8). Only a
guard holding the signing key can attenuate: the holder of a token cannot
mint one for themselves.

## Rotating the signing key

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

**Without a restart.** Pass `scope_keyring_provider=` (a callable returning
the current `SigningKeyring`, typically reading your secret manager) and
the guard re-reads it every `scope_key_reload_seconds`; each of the three
rotation steps becomes a write to the secret manager, picked up by every
process. A provider that raises leaves the last good keyring in place and
is logged once, so a secret-manager outage degrades to "no rotation right
now", never to "no signing". `reload_keys()` forces a read.

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
