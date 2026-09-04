# Security Policy

This is a security library, so it is worth being explicit about what
"a vulnerability in it" means, where to send one, and what will happen
next. Nothing here is a warranty; see the LICENSE for that.

## Reporting a vulnerability

**Do not open a public issue for a security report.** Use GitHub's private
vulnerability reporting on this repository
(Security → Report a vulnerability), which opens a private thread with the
maintainers.

If that is unavailable to you, email **matteo.cacciola@gmail.com** with
`llm-security-pipeline` in the subject line.

Useful things to include, in rough order of usefulness:

- What the guard does and what you believe it should do instead.
- A minimal reproduction: a string, a configuration, a sequence of calls.
  A failing test against `tests/` is ideal and is the fastest path to a
  fix.
- The version (`pip show llm-security-pipeline`) and which state backend
  you were using, if the finding involves shared state.
- Whether you intend to disclose publicly, and on what timeline.

You do not need to have a fix, and you do not need to be certain. A report
that turns out to be intended behaviour is a documentation bug worth
knowing about.

## What to expect

- **Acknowledgement within 5 working days.** This is a small project, not
  a vendor with an on-call rotation; if you have not heard back in that
  window, please chase.
- An assessment of whether the report is in scope (see below) and, if it
  is, a severity judgement stated in terms of what an attacker gets.
- A fix released as a new version, with the issue described in the release
  notes and credit to the reporter unless you ask otherwise.
- Coordinated disclosure: please give 90 days from acknowledgement before
  publishing, or less by agreement if a fix ships sooner.

## Scope

**In scope** — a defect in this library's own behaviour, for example:

- A guard reporting content as clean that it is documented to detect, or
  a bypass of a detector via encoding, normalization or input shaping.
- Redaction that leaves detected content in the output. (This has happened:
  `find_secrets` used to mishandle patterns with capture groups, so every
  occurrence after the first went out unredacted. See
  `tests/test_redaction.py`.)
- A capability token that verifies when it should not: forged or altered
  payload, expired, out of scope, presented by the wrong subject, or
  redeemed beyond `max_uses`.
- Loss of a cross-process guarantee: a rate limit or replay check that can
  be bypassed by racing multiple processes against any of the bundled
  backends.
- Resource exhaustion reachable from ordinary untrusted input — a
  pathological regex input, or an unbounded allocation.
- Anything that makes a failure silent when it should be loud. A guard
  that raises is a bug report; a guard that returns "clean" because it
  could not run is a vulnerability. This includes a degraded check that
  does not appear in `result.degraded` or in a `backend_degraded` audit
  event: fail-open is a supported configuration, fail-open that nobody can
  see is not.

**Out of scope** — real problems, but not defects in this code:

- **A jailbreak or injection phrasing the lexical heuristics miss.** They
  are documented as a supporting signal, not the defence; the structural
  `wrap_as_data` boundary is. New phrasings are welcome as a pull request
  against `config/patterns.json`, not as a security report.
- **Threshold tuning.** The defaults are chosen to be reasonable in
  general and are therefore wrong for any specific deployment. Measure
  yours with `llm_security_pipeline.evaluation` and change them. "The
  default blocked/failed to block my input" is a tuning question.
- **The in-memory stores not working across processes.** This is their
  documented behaviour, warned about in the README and in each class's
  docstring. Use a `StateBackend`.
- **An unauthenticated `session_id` being forgeable.** The library receives
  a principal, it cannot verify one; that boundary is deliberate and
  permanent. See the *Identity* section of the README.
- **Object-level authorization.** Whether this user may touch account 42 is
  your data model's question. The library carries and verifies signed
  constraints; it does not decide policy.
- Vulnerabilities in dependencies (report upstream, though do tell us if a
  version pin here is the reason you are exposed), or in your own
  integration of this library.

## Supported versions

Only the latest released minor version receives fixes. This is a 0.x
library and there is no long-term support branch.

## A standing caveat

This code has not been through an external security audit, and no
heuristic in it substitutes for red-teaming your own agent. Treat the
library as one layer among several, which is the only way any of this
works.
