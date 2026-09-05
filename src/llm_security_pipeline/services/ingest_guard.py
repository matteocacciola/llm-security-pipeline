"""
ingest_guard.py
Layer 0: the check that happens before anything is indexed.

`pre_process_external` scans retrieved content at query time, which is the
right last line of defence but the wrong place to catch poisoning. By then
the malicious chunk is already in the index, already being returned for
every semantically similar query, and already being scanned — and paid for
— on every single retrieval. Worse, retrieval-time scanning sees only the
top-k chunks that happen to come back; a document planted months ago that
surfaces for one specific question is never examined until the day it
works.

Ingestion is where the economics reverse: scan once, on the way in, with a
budget that would be unaffordable per-query, and keep a record of what was
decided.

Three things this adds over running the sanitizer at ingest time by hand:

* **A verdict, not a sanitization.** At query time the pipeline sanitizes
  and continues, because refusing to answer is a bad outcome. At ingest
  time nothing is waiting on the result, so the useful outputs are accept,
  quarantine (hold for review) and reject.
* **Trust tiers.** A curated internal wiki and a scraped forum thread do
  not deserve the same threshold. Per-source tiers let the same corpus
  hold both without either being mis-served.
* **Provenance.** The verdict and a content hash are recorded at ingest
  and re-checked at retrieval. That is what catches the case scanning
  cannot: content that was clean when indexed and was modified afterwards,
  in the vector store, by someone who already has write access to it.

Provenance needs somewhere to live; see `ProvenanceStore` in
sessions/stores.py. Verification is opt-in precisely because a store that
silently forgets would give false assurance.
"""

from __future__ import annotations

import dataclasses

import hashlib
from dataclasses import dataclass, field

from ..sessions.stores import ProvenanceRecord, ProvenanceStore
from ..resilience import (
    Degraded,
    PROVENANCE,
    FailurePolicy,
    ResilientBackend,
)
from .exfil_guard import ExfilGuard
from .sanitizer import SanitizationResult, Sanitizer

# Decisions
ACCEPT = "accept"
QUARANTINE = "quarantine"
REJECT = "reject"

# Trust tiers, in descending order of how much benefit of the doubt a
# source gets. The names are conventions, not an enum: add your own tiers
# by passing thresholds for them.
TRUSTED = "trusted"
PARTNER = "partner"
UNTRUSTED = "untrusted"

# (quarantine_at, reject_at) per tier. A trusted source has to look
# actively malicious before anything happens; an untrusted one is held for
# review on a much weaker signal, because the cost of quarantining a
# scraped page is close to zero and the cost of indexing a poisoned one is
# not.
# The untrusted tier quarantines on a single injection-phrase match (0.25
# on the sanitizer's scale). That is deliberately more sensitive than the
# pipeline's runtime threshold: an imperative addressed to a model is
# common enough in a chat message to need corroboration, and rare enough
# inside a document that claims to be reference material to be worth a
# human glance on its own.
DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    TRUSTED: (0.75, 0.95),
    PARTNER: (0.5, 0.8),
    UNTRUSTED: (0.25, 0.6),
}


def content_hash(content: str) -> str:
    """Stable identity for a piece of content. SHA-256 rather than a fast
    hash: this is compared against a value an attacker would like to
    forge."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class IngestVerdict:
    document_id: str
    source_id: str
    trust: str
    decision: str
    risk_score: float
    content_hash: str
    scan: SanitizationResult
    reasons: tuple[str, ...] = ()
    # In shadow mode `decision` is what was DONE (always accept, so the
    # document is indexed and retrievable) and `would_decide` is what
    # enforcement would have done. Equal to `decision` otherwise. Same
    # shape as blocked/would_block on the request path.
    would_decide: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == ACCEPT


@dataclass
class RetrievalVerdict:
    """Result of checking retrieved content against what was recorded at
    ingest."""

    document_id: str
    trusted: bool
    reason: str
    recorded: ProvenanceRecord | None = None

    @property
    def tampered(self) -> bool:
        return self.reason == "content_hash_mismatch"


@dataclass
class IngestGuard:
    """Scans documents on the way into an index and records what it decided.

    Example:
        guard = IngestGuard(provenance_store=backend.provenance_store)

        verdict = await guard.ingest(
            page_text, document_id="wiki:1024", source_id="scraped:example.com",
            trust=UNTRUSTED,
        )
        if verdict.accepted:
            index.add(page_text, metadata={"document_id": "wiki:1024"})
        else:
            review_queue.put(verdict)

    and at query time:

        check = await guard.verify_retrieved(chunk_text, document_id="wiki:1024")
        if check.tampered:
            ...
    """

    sanitizer: Sanitizer | None = None
    exfil_guard: ExfilGuard | None = None
    provenance_store: ProvenanceStore | None = None
    thresholds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    default_trust: str = UNTRUSTED
    failure_policy: FailurePolicy | None = None
    resilience: ResilientBackend | None = None

    def __post_init__(self) -> None:
        self.sanitizer = self.sanitizer or Sanitizer()
        self.exfil_guard = self.exfil_guard or ExfilGuard()
        # Provenance defaults to fail-closed. An unverifiable retrieval is
        # precisely the state this store exists to tell apart from a
        # verified one, so answering "verified" when the store is
        # unreachable would dissolve the distinction the feature is for.
        self.resilience = self.resilience or ResilientBackend(self.failure_policy)

    def _thresholds_for(self, trust: str) -> tuple[float, float]:
        try:
            return self.thresholds[trust]
        except KeyError:
            raise ValueError(
                f"No thresholds configured for trust tier {trust!r}. "
                f"Known tiers: {sorted(self.thresholds)}."
            ) from None

    # -- scanning --------------------------------------------------------

    def evaluate(
        self,
        content: str,
        *,
        document_id: str,
        source_id: str,
        trust: str | None = None,
    ) -> IngestVerdict:
        """Scan and decide, without recording anything. Pure and
        side-effect free, so it parallelizes across a process pool."""
        trust = trust or self.default_trust
        quarantine_at, reject_at = self._thresholds_for(trust)

        # Both are filled in by __post_init__; the fields stay Optional so
        # the constructor can take None to mean "use the default". Reading
        # them through locals states that invariant once instead of
        # scattering it, and keeps a type checker honest about the rest.
        sanitizer, exfil_guard = self.sanitizer, self.exfil_guard
        if sanitizer is None or exfil_guard is None:  # pragma: no cover - set in __post_init__
            raise RuntimeError("IngestGuard was constructed without running __post_init__.")

        scan = sanitizer.scan_text(
            content, threshold=quarantine_at, tag="EXTERNAL_CONTENT", source_id=source_id,
        )
        reasons: list[str] = []
        score = scan.risk_score

        if scan.matched_patterns:
            reasons.append("injection_phrases")
        if scan.decoded_payload_hits:
            reasons.append("encoded_payload")
        if scan.hidden_text_hits:
            reasons.append("hidden_text")

        # A document carrying a beacon URL is a poisoning vector even when
        # it contains no instructions at all: the model will happily
        # reproduce the image markup into an answer, and the fetch happens
        # in the user's client.
        exfil = exfil_guard.scan(content)
        if exfil.suspicious_findings:
            reasons.append("exfil_url")
            score = max(score, 0.8 if exfil.blocked else 0.5)

        decision = ACCEPT
        if score >= reject_at:
            decision = REJECT
        elif score >= quarantine_at:
            decision = QUARANTINE

        return IngestVerdict(
            document_id=document_id,
            source_id=source_id,
            trust=trust,
            decision=decision,
            risk_score=round(score, 2),
            content_hash=content_hash(content),
            scan=scan,
            reasons=tuple(reasons),
            would_decide=decision,
        )

    # -- provenance ------------------------------------------------------

    def _resilience(self) -> ResilientBackend:
        if self.resilience is None:  # pragma: no cover - set in __post_init__
            raise RuntimeError("IngestGuard was constructed without running __post_init__.")
        return self.resilience

    # -- review queue ----------------------------------------------------
    # "Quarantine" was a verdict with nowhere to go: the record said a
    # human should look, and nothing let the human look, decide, or make
    # the decision stick. These are that. A decision made here is written
    # over the provenance record, so verify_retrieved sees it on the next
    # retrieval — approving a document is what makes it retrievable.

    async def review_queue(self, limit: int = 100) -> list[ProvenanceRecord]:
        """Quarantined documents, oldest first."""
        store = self._require_store()
        result = await self._resilience().run(
            PROVENANCE, lambda: store.list_by_decision(QUARANTINE, limit),
        )
        return [] if isinstance(result, Degraded) else result

    async def approve(self, document_id: str) -> bool:
        """A reviewer read it and it is fine: retrievable from now on."""
        return await self._decide(document_id, ACCEPT)

    async def reject(self, document_id: str) -> bool:
        """A reviewer read it and it is not: stays in the index if you
        leave it there, but never verifies at retrieval."""
        return await self._decide(document_id, REJECT)

    async def _decide(self, document_id: str, decision: str) -> bool:
        store = self._require_store()
        result = await self._resilience().run(
            PROVENANCE, lambda: store.set_decision(document_id, decision),
        )
        return False if isinstance(result, Degraded) else bool(result)

    def _require_store(self) -> ProvenanceStore:
        if self.provenance_store is None:
            raise RuntimeError(
                "IngestGuard needs a provenance_store to record or verify "
                "ingest decisions. Pass one explicitly (RedisStateBackend "
                "provides one; InMemoryProvenanceStore is fine for a single "
                "process). Verification is not enabled by default because a "
                "store that quietly forgets is worse than no store at all."
            )
        return self.provenance_store

    async def ingest(
        self,
        content: str,
        *,
        document_id: str,
        source_id: str,
        trust: str | None = None,
        record_rejected: bool = True,
        shadow: bool = False,
    ) -> IngestVerdict:
        """Evaluate and record the verdict against `document_id`.

        With `shadow=True` the real verdict is computed and reported in
        `would_decide`, but the document is recorded as accepted: a dry
        run of the ingest thresholds on a real corpus, the way shadow mode
        is a dry run of the request thresholds on real traffic. Nothing is
        quarantined, so the review queue stays empty and every document
        stays retrievable; the audit log and metrics say what would have
        happened. Turn it off before trusting the queue.
        """
        verdict = self.evaluate(
            content, document_id=document_id, source_id=source_id, trust=trust,
        )
        if shadow:
            verdict = self.shadowed(verdict)
        store = self._require_store()
        if verdict.accepted or record_rejected:
            record = ProvenanceRecord(
                document_id=verdict.document_id,
                content_hash=verdict.content_hash,
                source_id=verdict.source_id,
                trust=verdict.trust,
                decision=verdict.decision,
                risk_score=verdict.risk_score,
            )
            # A verdict that is not recorded is a document that will read as
            # "never ingested" forever after, so under a closed policy the
            # caller is told rather than left to index it.
            await self._resilience().run(PROVENANCE, lambda: store.record(record))
        return verdict

    @staticmethod
    def shadowed(verdict: IngestVerdict) -> IngestVerdict:
        """The verdict as shadow mode records it: accepted, with what
        enforcement would have done kept in `would_decide`."""
        return dataclasses.replace(verdict, decision=ACCEPT, would_decide=verdict.decision)

    async def verify_retrieved(self, content: str, *, document_id: str) -> RetrievalVerdict:
        """Check retrieved content against the record made at ingest.

        Three ways this comes back untrusted, and they are worth
        distinguishing in your logs: the document was never ingested
        through this path at all, it was ingested and not accepted, or its
        bytes have changed since. Only the last one is tampering; the first
        is usually a pipeline that bypassed the guard.
        """
        store = self._require_store()
        record = await self._resilience().run(PROVENANCE, lambda: store.get(document_id))
        if isinstance(record, Degraded):
            # Only reachable under an open policy, which for provenance
            # means "treat unverifiable as verified". Rarely the right
            # choice; the reason is recorded so it does not read as a
            # successful verification.
            return RetrievalVerdict(document_id, True, "provenance_unavailable")
        if record is None:
            return RetrievalVerdict(document_id, False, "no_provenance_record")
        if record.content_hash != content_hash(content):
            return RetrievalVerdict(document_id, False, "content_hash_mismatch", record)
        if record.decision != ACCEPT:
            return RetrievalVerdict(document_id, False, f"ingested_as_{record.decision}", record)
        return RetrievalVerdict(document_id, True, "verified", record)
