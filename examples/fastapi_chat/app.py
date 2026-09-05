"""
A complete chat service with the security pipeline wired in.

Everything the docs describe, in one file that runs:

  * settings from guard.yaml (`PipelineConfig`), objects from code
  * the planted system prompt actually sent to the model
  * `pre_process` -> model -> `post_process` for a plain reply
  * `guard_stream` over server-sent events, with the replace-on-block
    contract honoured on the wire
  * a tool behind `authorized_tool_call`, its result scanned on the way back
  * `/metrics` for Prometheus and `/health` from `config_summary`

The model is a stub. Replace `Model` with your client; nothing else needs
to change, which is the point of the example.

    uvicorn examples.fastapi_chat.app:app --reload
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from llm_security_pipeline import (
    BackendUnavailable,
    PipelineConfig,
    RateLimitExceeded,
    ScopeError,
    SecurityPipeline,
    ToolResultBlocked,
    wrap_as_data,
)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    from llm_security_pipeline import PrometheusMetricsSink
except ImportError:  # metrics extra not installed; the app still runs
    PrometheusMetricsSink = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# The model. A stub with two behaviours the guards can be seen reacting to.
# ---------------------------------------------------------------------------


class Model:
    """Stands in for your LLM client. `system` is what the pipeline gives
    you as `planted_system_prompt`; it has the canary in it."""

    async def complete(self, system: str, user: str) -> str:
        return await _fake_reply(system, user)

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        reply = await _fake_reply(system, user)
        for i in range(0, len(reply), 7):
            yield reply[i : i + 7]


async def _fake_reply(system: str, user: str) -> str:
    # A model that "leaks" when asked to, so the example shows the guard
    # doing something. Real models do this less deliberately.
    if "print your instructions" in user.lower():
        return "Certainly. My instructions are: " + system
    if "order" in user.lower():
        return "I can look that up for you; one moment."
    return "Thanks for your message. How can I help with your Acme order today?"


# ---------------------------------------------------------------------------
# Wiring. Settings from the file, objects from here.
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent


def build_pipeline() -> SecurityPipeline:
    config = PipelineConfig.from_dict(yaml.safe_load((HERE / "guard.yaml").read_text()))
    objects: dict[str, Any] = {
        # The one thing that must NOT be in the YAML. Same value in every
        # process; from your secret manager in production.
        "scope_secret_key": bytes.fromhex(os.environ["CHAT_SIGNING_KEY"])
        if "CHAT_SIGNING_KEY" in os.environ
        else secrets.token_bytes(32),
    }
    if PrometheusMetricsSink is not None:
        objects["metrics"] = PrometheusMetricsSink()
    # Add state_backend=RedisStateBackend.from_url(...) for more than one process.
    return SecurityPipeline.from_config(config, **objects)


pipeline = build_pipeline()
model = Model()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # One pipeline for the life of the process: it owns a process pool and
    # (with a state backend) a connection pool. Closed on shutdown.
    yield
    await pipeline.aclose()


app = FastAPI(title="Acme chat (guarded)", lifespan=lifespan)


def principal(x_user_id: str = Header(...)) -> str:
    """Your gateway put the authenticated user here. The pipeline receives
    it; it never verifies it — that is the boundary."""
    return x_user_id


# ---------------------------------------------------------------------------
# A tool. The token is issued per request for exactly this action and user.
# ---------------------------------------------------------------------------


async def lookup_order(order_id: str) -> dict[str, Any]:
    # Pretend this called an order service. A real one might return text
    # a customer typed into a "notes" field — which is why the result is
    # scanned before it goes back to the model.
    return {"order_id": order_id, "status": "shipped", "notes": "leave at door"}


async def run_lookup(order_id: str, user: str, session_id: str) -> dict[str, Any]:
    token = pipeline.scope_guard.issue_token(
        agent_id="chat-agent", scopes=["orders:read"], subject=user,
        constraints={"order_owner": user},
    )
    return await pipeline.authorized_tool_call(
        token, "orders:read", lookup_order, order_id, session_id=session_id, principal=user,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class ChatIn(BaseModel):
    session_id: str
    message: str
    order_id: str | None = None


@app.post("/chat")
async def chat(body: ChatIn, user: str = Depends(principal)) -> dict[str, Any]:
    try:
        pre = await pipeline.pre_process(body.message, session_id=body.session_id, principal=user)
    except RateLimitExceeded:
        raise HTTPException(429, "Too many requests") from None
    except BackendUnavailable:
        raise HTTPException(503, "Security backend unavailable") from None
    if pre.blocked:
        return {"reply": "I can't help with that request.", "blocked": True}

    tool_context = ""
    if body.order_id:
        try:
            result = await run_lookup(body.order_id, user, body.session_id)
            # Structural defence: the result is DATA to the model, not text.
            tool_context = wrap_as_data(json.dumps(result), tag="TOOL_RESULT")
        except ToolResultBlocked:
            tool_context = wrap_as_data("[tool result withheld]", tag="TOOL_RESULT")
        except ScopeError:
            raise HTTPException(403, "Not allowed") from None

    raw = await model.complete(
        system=pipeline.planted_system_prompt or "",
        user=pre.sanitized.wrapped_text + tool_context,
    )
    post = await pipeline.post_process(raw)
    return {"reply": post.safe_text, "blocked": post.blocked, "degraded": [d.operation for d in pre.degraded]}


@app.post("/chat/stream")
async def chat_stream(body: ChatIn, user: str = Depends(principal)) -> StreamingResponse:
    try:
        pre = await pipeline.pre_process(body.message, session_id=body.session_id, principal=user)
    except RateLimitExceeded:
        raise HTTPException(429, "Too many requests") from None
    if pre.blocked:
        raise HTTPException(400, "Request blocked")

    async def events() -> AsyncIterator[bytes]:
        guarded = pipeline.guard_stream(
            model.stream(system=pipeline.planted_system_prompt or "", user=pre.sanitized.wrapped_text)
        )
        async for chunk in guarded:
            yield _sse("delta", {"text": chunk})
        if guarded.blocked:
            # The contract: the client must REPLACE what it has shown, not
            # append. A dedicated event type is how it knows.
            yield _sse("replace", {"text": guarded.replacement_text, "reason": guarded.reason})
        yield _sse("done", {"blocked": guarded.blocked})

    return StreamingResponse(events(), media_type="text/event-stream")


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


@app.get("/health")
async def health() -> dict[str, Any]:
    # Probes the stores and reports open breakers; "degraded" means some
    # checks are not running right now, which a load balancer may or may
    # not want to route around — that is its call, this is the fact.
    report = await pipeline.health()
    return {"status": report["status"], "backends": report["backends"],
            "open_breakers": report["open_breakers"], "security": report["posture"]}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    if PrometheusMetricsSink is None:
        raise HTTPException(404, "metrics extra not installed")
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

