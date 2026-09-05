"""
Mutation check for the security decisions.

A test suite that is green proves nothing about what it would catch. This
breaks one security decision at a time — the expiry check, the scope
check, the hold-back, the fail-closed policy — runs the unit suite, and
reports whether any test noticed. A mutant that SURVIVES is a decision no
test protects, which for this library is a finding in its own right.

Not a CI gate: it runs the suite once per mutation (a couple of minutes),
and a green run is a floor, not a ceiling. Run it when you touch a guard:

    uv run python tools/mutate.py
    uv run python tools/mutate.py --only revocation   # substring filter

Each mutation names an exact source line, so a refactor that moves the
line shows up as ANCHOR-MISSING rather than as a silently skipped check;
update the anchor, do not delete the entry.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = "src/llm_security_pipeline"

# (file, original line(s), mutated line(s), what the mutation breaks)
MUTATIONS: list[tuple[str, str, str, str]] = [
    (f"{SRC}/services/output_guard.py",
     "def _apply_spans(text: str, spans: list[_Span]) -> str:\n    out = text",
     "def _apply_spans(text: str, spans: list[_Span]) -> str:\n    return text\n    out = text",
     "redaction does nothing"),
    (f"{SRC}/services/output_guard.py",
     "            spans.append(_Span(start, start + len(canary.token), CANARY_CATEGORY, priority=0))",
     "            pass", "canary never matched"),
    (f"{SRC}/services/output_guard.py",
     "                spans.append(_Span(start, start + len(literal), CROSS_TENANT_CATEGORY, priority=0))",
     "                pass", "cross-tenant literal never matched"),
    (f"{SRC}/services/scope_guard.py",
     '        if now > token.payload["expires_at"]:\n            raise ScopeError("Token expired.")',
     '        if False:\n            raise ScopeError("Token expired.")', "expiry never checked"),
    (f"{SRC}/services/scope_guard.py",
     '        if action not in token.payload["scopes"]:',
     '        if False and action not in token.payload["scopes"]:', "scope never checked"),
    (f"{SRC}/services/scope_guard.py",
     '        if uses > token.payload["max_uses"]:',
     '        if False and uses > token.payload["max_uses"]:', "replay never checked"),
    (f"{SRC}/services/scope_guard.py",
     "        if not hmac.compare_digest(expected, str(token.signature)):",
     "        if False and not hmac.compare_digest(expected, str(token.signature)):",
     "signature never checked"),
    (f"{SRC}/services/scope_guard.py",
     '                and float(token.payload.get("issued_at", 0.0)) <= revoked',
     '                and float(token.payload.get("issued_at", 0.0)) > revoked', "revocation inverted"),
    (f"{SRC}/services/scope_guard.py",
     "        if widening:\n            raise ScopeError(",
     "        if False:\n            raise ScopeError(", "attenuation can widen scope"),
    (f"{SRC}/services/scope_guard.py",
     "            if not hmac.compare_digest(str(presented), self._audience):",
     "            if False:", "audience never checked"),
    (f"{SRC}/services/rate_limiter.py",
     "        if count > limits.max_requests_per_window:\n            raise RateLimitExceeded(",
     "        if False:\n            raise RateLimitExceeded(", "request budget never enforced"),
    (f"{SRC}/services/sanitizer.py",
     "        if self.max_scan_chars is not None and len(text) > self.max_scan_chars:",
     "        if False:", "input size cap off"),
    (f"{SRC}/services/sanitizer.py",
     "    return _WORD.sub(fix, text), folded", "    return text, 0", "confusables never folded"),
    (f"{SRC}/services/sanitizer.py",
     "        if hidden_hits:\n            # Weighted", "        if False:\n            # Weighted",
     "hidden text not weighted"),
    (f"{SRC}/services/streaming_guard.py",
     "        cut = len(window) if final else max(offset, len(window) - self.holdback_chars)",
     "        cut = len(window)", "streaming hold-back off"),
    (f"{SRC}/services/streaming_guard.py",
     "        self._leaked = straddles and self._emitted_len > 0",
     "        self._leaked = False", "partial leak never reported"),
    (f"{SRC}/resilience.py",
     "        if decision == FAIL_CLOSED:\n            raise BackendUnavailable(operation, cause)",
     "        if False:\n            raise BackendUnavailable(operation, cause)", "fail-closed becomes fail-open"),
    (f"{SRC}/services/detectors.py",
     "            if registration.mode == ENFORCING:\n                result.enforcing_score = max(result.enforcing_score, contribution)",
     "            if registration.mode == ENFORCING:\n                result.advisory_score = max(result.advisory_score, contribution)",
     "enforcing detectors never enforce"),
    (f"{SRC}/pipeline.py",
     '            if self.enforcement == "enforce":\n                raise ToolResultBlocked(action, scan, output)',
     "            if False:\n                raise ToolResultBlocked(action, scan, output)", "tool results never blocked"),
    (f"{SRC}/pipeline.py",
     '        stamped = {"schema_version": AUDIT_SCHEMA_VERSION, **data}',
     "        stamped = dict(data)", "schema_version missing from audit"),
    (f"{SRC}/metrics.py",
     "        return {k: str(v) for k, v in labels.items() if k not in FORBIDDEN_LABELS}",
     "        return {k: str(v) for k, v in labels.items()}", "unknown-metric path lets identifiers through"),
    (f"{SRC}/tracing.py",
     "    if key in FORBIDDEN_LABELS:", "    if False:", "span attributes let identifiers through"),
    (f"{SRC}/services/exfil_guard.py",
     '        if auto_fetch and hard_hit:\n            severity = "block"',
     '        if False:\n            severity = "block"', "exfil beacons never block"),
    (f"{SRC}/services/ingest_guard.py",
     "        if record is None:", "        if False:", "missing provenance treated as present"),
    (f"{SRC}/sessions/stores.py",
     "        while events and events[0] <= cutoff:\n            events.popleft()",
     "        pass", "sliding window never slides"),
]


def run(only: str | None) -> int:
    survivors = 0
    for path, original, mutated, what in MUTATIONS:
        if only and only not in what:
            continue
        target = ROOT / path
        source = target.read_text()
        if original not in source:
            print(f"ANCHOR-MISSING  {what}  ({path})")
            survivors += 1
            continue
        target.write_text(source.replace(original, mutated, 1))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "-x", "-p", "no:cacheprovider",
                 "-o", "addopts=", "-m", "not integration and not perf"],
                capture_output=True, text=True, cwd=ROOT, timeout=900,
            )
        finally:
            target.write_text(source)
        killed = result.returncode != 0
        survivors += 0 if killed else 1
        print(f"{'killed' if killed else 'SURVIVED':15s} {what}")
    print(f"\n{survivors} surviving mutant(s)")
    return 1 if survivors else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="run only mutations whose description contains this substring")
    sys.exit(run(parser.parse_args().only))
