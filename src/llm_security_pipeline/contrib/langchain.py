"""
LangChain integration.

Three pieces, each a plain Runnable or wrapper, composable with `|`:

    from llm_security_pipeline.contrib.langchain import (
        GuardedChatModel, guard_retriever, guard_tool,
    )

    model     = GuardedChatModel(pipeline, ChatOpenAI())      # pre_process + post_process, streaming too
    retriever = guard_retriever(pipeline, vector_store.as_retriever())   # pre_process_external_batch
    tool      = guard_tool(pipeline, my_tool, scopes=["crm:read"])       # authorized_tool_call

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

--- What is guarded where --------------------------------------------------

`GuardedChatModel` sits in front of any chat model. The last human
message goes through `pre_process`; if it is blocked the model is never
called and the reply is the refusal. The system message is replaced by
the pipeline's *planted* system prompt when a canary is configured. The
model's reply goes through `post_process` (or `guard_stream` when
streamed), so secrets, the canary and cross-tenant identifiers are caught
on the way out. Session and principal come from the call's `config`
(`configurable.session_id`, `configurable.principal`), which is how
LangChain threads per-request identity through a chain.

`guard_retriever` wraps a retriever: every returned document goes through
`pre_process_external_batch`, blocked documents are dropped, and — when
documents carry a `document_id` in metadata and a provenance store is
configured — `verify_retrieved` drops the ones that do not verify. The
surviving page content is wrapped as data, which is the structural
defence.

`guard_tool` wraps a tool so every invocation runs behind
`authorized_tool_call` with a token issued for exactly that action and
principal: budget, scope, replay, and the result scanned on its way back.

--- Why not a callback handler --------------------------------------------

LangChain callbacks observe; they cannot rewrite an input or withhold an
output. A guard that can only watch is an audit log, and the pipeline
already has one of those. These wrappers sit *in* the chain, where a
blocked input stops the model from being called at all.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel, LanguageModelInput
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import RunnableConfig, RunnableLambda
except ImportError as exc:  # pragma: no cover - depends on the extra
    raise ImportError(
        "llm_security_pipeline.contrib.langchain needs langchain-core. Install it with "
        "pip install 'llm-security-pipeline[langchain]'."
    ) from exc

from ..pipeline import SecurityPipeline, ToolResultBlocked
from ..services.rate_limiter import RateLimitExceeded
from ..services.sanitizer import wrap_as_data

BLOCKED_REPLY = "I can't help with that request."


def _identity(config: RunnableConfig | None) -> tuple[str | None, str | None]:
    """session_id and principal from the run's `configurable` block."""
    conf = (config or {}).get("configurable", {}) or {}
    return conf.get("session_id"), conf.get("principal")


class GuardedChatModel(BaseChatModel):
    """A chat model with the pipeline in front of and behind it.

    Subclassing BaseChatModel rather than wrapping with a RunnableLambda
    keeps the result a chat model: `.bind_tools`, `.with_structured_output`
    and every consumer that checks `isinstance(x, BaseChatModel)` keep
    working. Only the async paths are guarded — the pipeline is async —
    and the sync ones raise, loudly, rather than run unguarded.
    """

    pipeline: Any
    model: Any

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return f"guarded({getattr(self.model, '_llm_type', type(self.model).__name__)})"

    # -- sync: refused, not silently unguarded ---------------------------

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise RuntimeError(
            "GuardedChatModel guards the async paths only (the pipeline is async). "
            "Use ainvoke/astream, or asyncio.run(...) around a sync call site."
        )

    # -- async ------------------------------------------------------------

    def _prepare(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage], HumanMessage | None]:
        """Swap in the planted system prompt; find the message to scan."""
        planted = self.pipeline.planted_system_prompt
        out: list[BaseMessage] = []
        saw_system = False
        for message in messages:
            if isinstance(message, SystemMessage) and planted is not None and not saw_system:
                out.append(SystemMessage(content=planted))
                saw_system = True
            else:
                out.append(message)
        if planted is not None and not saw_system:
            out.insert(0, SystemMessage(content=planted))
        last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        return out, last_human

    async def _pre(self, messages: list[BaseMessage], config: RunnableConfig | None):
        session_id, principal = _identity(config)
        prepared, last_human = self._prepare(messages)
        if last_human is None:
            return prepared, None, principal
        text = last_human.content if isinstance(last_human.content, str) else str(last_human.content)
        pre = await self.pipeline.pre_process(text, session_id=session_id, principal=principal)
        if pre.blocked:
            return prepared, pre, principal
        # The model sees the sanitized, data-wrapped text, not the raw one.
        prepared = [
            HumanMessage(content=pre.sanitized.wrapped_text) if m is last_human else m for m in prepared
        ]
        return prepared, pre, principal

    async def _agenerate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs,
    ) -> ChatResult:
        config = kwargs.pop("config", None)
        prepared, pre, principal = await self._pre(messages, config)
        if pre is not None and pre.blocked:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=BLOCKED_REPLY))])
        reply = await self.model.ainvoke(prepared, config=config, stop=stop, **kwargs)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        post = await self.pipeline.post_process(text, principal=principal)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=post.safe_text))])

    async def _astream(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs,
    ) -> AsyncIterator[ChatGenerationChunk]:
        config = kwargs.pop("config", None)
        prepared, pre, principal = await self._pre(messages, config)
        if pre is not None and pre.blocked:
            yield ChatGenerationChunk(message=AIMessageChunk(content=BLOCKED_REPLY))
            return

        async def source() -> AsyncIterator[str]:
            async for chunk in self.model.astream(prepared, config=config, stop=stop, **kwargs):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if content:
                    yield content

        guarded = self.pipeline.guard_stream(source(), principal=principal)
        async for text in guarded:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        if guarded.blocked:
            # The replace-on-block contract cannot be expressed in a token
            # stream: what was streamed is out. The refusal is appended and
            # the block is flagged in response metadata, so a UI that
            # renders metadata can replace, and one that cannot at least
            # shows the refusal.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=f"\n{guarded.replacement_text}",
                    response_metadata={"llm_security_blocked": True, "llm_security_reason": guarded.reason},
                )
            )

    # LangChain calls _generate/_agenerate through generate_prompt with
    # the config already consumed; ainvoke on a chat model goes via
    # agenerate_prompt(..., **kwargs) where the run config is not passed
    # down. Override ainvoke/astream so the config reaches _pre.
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        messages = self._convert_input(input).to_messages()
        result = await self._agenerate(messages, stop=stop, config=config, **kwargs)
        message = result.generations[0].message
        return message if isinstance(message, AIMessage) else AIMessage(content=str(message.content))

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        messages = self._convert_input(input).to_messages()
        async for chunk in self._astream(messages, stop=stop, config=config, **kwargs):
            message = chunk.message
            yield message if isinstance(message, AIMessageChunk) else AIMessageChunk(content=str(message.content))


def guard_retriever(pipeline: SecurityPipeline, retriever: BaseRetriever):
    """A retriever whose documents are scanned, verified and data-wrapped."""

    async def guarded(query: str, config: RunnableConfig | None = None) -> list[Document]:
        session_id, principal = _identity(config)
        documents = await retriever.ainvoke(query, config=config)
        if not documents:
            return []
        chunks = [
            (doc.page_content, str(doc.metadata.get("document_id") or doc.metadata.get("source") or f"doc:{i}"))
            for i, doc in enumerate(documents)
        ]
        scans = await pipeline.pre_process_external_batch(chunks, session_id=session_id, principal=principal)
        kept: list[Document] = []
        for doc, scan in zip(documents, scans, strict=True):
            if scan.blocked:
                continue
            document_id = doc.metadata.get("document_id")
            if document_id and pipeline.ingest_guard.provenance_store is not None:
                verdict = await pipeline.verify_retrieved(doc.page_content, document_id=str(document_id))
                if not verdict.trusted:
                    continue
            kept.append(Document(
                page_content=wrap_as_data(scan.normalized_text, tag="RETRIEVED_DOCUMENT"),
                metadata={**doc.metadata, "llm_security_risk_score": scan.risk_score},
            ))
        return kept

    return RunnableLambda(guarded)


def guard_tool(pipeline: SecurityPipeline, tool: Any, scopes: list[str], action: str | None = None):
    """A tool that runs behind authorized_tool_call.

    `tool` is anything with a name and an `ainvoke`/`arun`/callable; the
    action defaults to its name. A token is issued per invocation for that
    action and the caller's principal, so a leaked token buys one call of
    one tool for one user.
    """
    name = action or getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
    if name not in scopes:
        raise ValueError(f"action {name!r} must be one of the scopes the token is issued with: {scopes}")

    async def call(tool_input: Any) -> Any:
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(tool_input)
        outcome = tool(tool_input)
        return await outcome if inspect.isawaitable(outcome) else outcome

    async def guarded(tool_input: Any, config: RunnableConfig | None = None) -> Any:
        session_id, principal = _identity(config)
        token = pipeline.scope_guard.issue_token("langchain", scopes, subject=principal)
        try:
            return await pipeline.authorized_tool_call(
                token, name, call, tool_input, session_id=session_id, principal=principal,
            )
        except ToolResultBlocked as exc:
            return {"error": "tool result withheld by the security layer", "action": exc.action}
        except RateLimitExceeded:
            return {"error": "tool call budget exceeded", "action": name}

    return RunnableLambda(guarded)

