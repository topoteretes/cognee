"""The real chat-memory adapter, backed by cognee.

This satisfies the same contract as the fake adapter and resolves scope through
the same per_user_scope policy, so the bot logic proven offline is the bot
logic that runs live.

Two design notes worth reading before you trust the citation path:

1. Memory boundary. ingest writes with dataset_name=brain:{canonical_user} and
   session_id={transport}:{source}. Durable recall targets the whole brain
   dataset, so a note from Telegram last week is recalled from web today; the
   session only gives fast recent context.

2. Citations. cognee's recall(include_references=True) grounds its Evidence
   block in document chunks via term overlap, NOT in the external_metadata
   deeplink stamped at ingest (verified against
   cognee/modules/retrieval/utils/references.py). So to cite the original
   transport message, this adapter keeps its own ingest-time citation map, the
   approach the #3608 design calls for ("the adapter records a cognee source to
   platform message ref mapping at ingest"). include_references is still passed
   so cognee's own grounding enriches the answer text. The external_metadata
   stamp and the deterministic data_id are kept for citations and for a future
   dedup-aware partial forget (see the TODO on forget()); forget in this version
   is whole-brain only.

cognee is imported lazily inside each method so importing this module (and the
scope smoke test) needs neither cognee installed nor any API key.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Optional, Union

from .interface import Answer, ChatMemoryAdapter, Citation, Conversation, Message, Scope
from .scope_policy import per_user_scope, resolve_user

# Stable namespace so a given (transport, source, ts) always maps to the same
# data_id. Idempotent ingest, and resolvable for per-message forget.
_DATA_ID_NAMESPACE = uuid.UUID("b1c2d3e4-f5a6-4b7c-8d9e-0a1b2c3d4e5f")

# Shown when the brain has nothing relevant (or no dataset yet).
_EMPTY_MEMORY = "I do not have anything in your memory about that yet."

# cognee renders grounded references under this header (see
# cognee/modules/retrieval/utils/references.py). Each bullet names the retrieved
# chunk's data_id, which we match back to the ingest-time citation map.
_EVIDENCE_HEADER = "Evidence:"
_EVIDENCE_DATA_ID_RE = re.compile(
    r"data_id:\s*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


@dataclass
class _CitationRecord:
    text: str
    transport: str
    source: str
    ts: str
    deeplink: str
    data_id: uuid.UUID
    dataset_id: Optional[str] = None


class CogneeChatMemoryAdapter(ChatMemoryAdapter):
    def __init__(self, top_k: int = 15, use_graph_completion: bool = True) -> None:
        self._top_k = top_k
        self._use_graph_completion = use_graph_completion
        # dataset name -> ingest-time citation records (the source-to-message map)
        self._citations: dict[str, list[_CitationRecord]] = {}

    def scope(self, conversation: Conversation) -> Scope:
        return per_user_scope(conversation)

    async def ingest(self, conversation: Conversation, message: Message) -> None:
        from ..config import load_cognee_env

        load_cognee_env()
        import cognee
        from cognee.tasks.ingestion.data_item import DataItem

        scope = self.scope(conversation)
        canonical = resolve_user(conversation)
        data_id = uuid.uuid5(
            _DATA_ID_NAMESPACE, f"{conversation.transport}:{conversation.source}:{message.ts}"
        )
        item = DataItem(
            data=message.text,
            data_id=data_id,
            external_metadata={
                "transport": conversation.transport,
                "source": conversation.source,
                "canonical_user": canonical,
                "ts": message.ts,
                "deeplink": message.deeplink or conversation.msg_ref or "",
            },
        )
        # Ingest writes dataset-only. The session field is resolved (scope.session)
        # and stays in the Scope contract for #3608 compatibility, but this
        # reference adapter deliberately does NOT write to the session cache,
        # because under access-control-off in this single-user config cognee's
        # session->graph distillation bridge fails with a 422 (the background
        # improve task runs as a user that "does not have write access to dataset:
        # brain:..."), which strands the note in the session cache and never
        # reaches the durable graph. Diagnosed with isolated probes: dataset-only
        # ingest reaches the graph and recalls cleanly; session ingest does not.
        # A session-cache-backed adapter (CACHING on, or the merged #3608 adapter)
        # would populate scope.session instead. self_improvement=False because the
        # dataset path does not need the improve enrichment and skipping it avoids
        # needless latency (dataset-only ingest throws no 422 either way).
        result = await cognee.remember(
            item,
            dataset_name=scope.dataset,
            run_in_background=True,
            self_improvement=False,
        )
        self._citations.setdefault(scope.dataset, []).append(
            _CitationRecord(
                text=message.text,
                transport=conversation.transport,
                source=conversation.source,
                ts=message.ts,
                deeplink=message.deeplink or conversation.msg_ref or "",
                data_id=data_id,
                dataset_id=getattr(result, "dataset_id", None),
            )
        )

    async def answer(self, conversation: Conversation, query: str) -> Answer:
        from ..config import load_cognee_env

        load_cognee_env()
        import cognee

        from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError

        scope = self.scope(conversation)
        # No session_id on recall either: this adapter is dataset-only (see ingest).
        kwargs = dict(
            datasets=[scope.dataset],
            include_references=True,
            top_k=self._top_k,
        )
        if self._use_graph_completion:
            # Real multi-hop traversal across transports, the cross-source demo moment.
            from cognee.modules.search.types.SearchType import SearchType

            kwargs["query_type"] = SearchType.GRAPH_COMPLETION

        try:
            results = await cognee.recall(query, **kwargs)
        except DatasetNotFoundError:
            # No dataset yet, or it was just wiped by forget. Empty memory, not an error.
            return Answer(text=_EMPTY_MEMORY)

        raw = "\n".join(_render_result(item) for item in results).strip()
        if not raw:
            return Answer(text=_EMPTY_MEMORY)

        answer_text, evidence = _split_evidence(raw)
        if not answer_text:
            return Answer(text=_EMPTY_MEMORY)

        # Cite structurally, not by prose-matching a refusal. A note is cited only
        # if BOTH hold: cognee actually retrieved it (its data_id appears in the
        # Evidence block) AND the answer uses content from that note beyond the
        # query's own words. A graph-completion refusal only echoes the query
        # subject (which is why an answer-grounded Evidence block can still name
        # the note during the post-ingest race where the chunk exists but the
        # graph is not built): it contributes no distinctive term, so it is never
        # cited, regardless of how the refusal is phrased. See select_citations.
        citations = select_citations(
            self._citations.get(scope.dataset, []), evidence, answer_text, query
        )

        # Cosmetic only: tidy an obviously-refusing, already-uncited answer into the
        # friendly empty message. Correctness (no false citation) does NOT depend on
        # this prose check; the structural guard above already guarantees it.
        if not citations and _is_no_answer(answer_text):
            return Answer(text=_EMPTY_MEMORY)
        return Answer(text=answer_text, citations=citations)

    async def forget(self, target: Union[Conversation, str]) -> None:
        """Whole-brain forget.

        Wipes the entire per-user brain in one call. Safe because the brain is
        single-user: there is no other user's data in the dataset to orphan.
        The caller (the bot) also drops the user's identity links so no
        transport silently re-attaches.
        """
        from ..config import load_cognee_env

        load_cognee_env()
        import cognee

        from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError

        if isinstance(target, Conversation):
            dataset = self.scope(target).dataset
        else:
            dataset = f"brain:{target}"
        try:
            await cognee.forget(dataset=dataset)
        except (DatasetNotFoundError, AttributeError):
            # Idempotent wipe: the brain may never have been created (/forget me
            # before any capture) or was already wiped. cognee resolves an unknown
            # dataset name to None and dereferences .id, raising AttributeError
            # (not DatasetNotFoundError), so both mean "already empty". Kept narrow,
            # never `except Exception`, so a real failure on a destructive command
            # is not silently swallowed.
            pass
        self._citations.pop(dataset, None)

    # TODO(follow-up, #3608 adapter core): per-transport / selective forget
    # (drop only one transport's captures, e.g. "forget just my Telegram
    # notes") is intentionally NOT implemented here. After cognify merges facts
    # from different transports into shared nodes within one brain, naive subset
    # deletion by external_metadata["transport"] can orphan shared facts that
    # other captures still reference. A correct implementation needs
    # deduplication-aware deletion: remove a node/edge only when no other data
    # references it. That belongs in the #3608 adapter core, not in this bot, so
    # it is deferred. The ingest-time external_metadata stamp (transport,
    # source, canonical_user, ts, deeplink) and the deterministic data_id are
    # retained precisely so that future dedup-aware partial forget has what it
    # needs.


def select_citations(
    records: "list[_CitationRecord]", evidence: str, answer_text: str, query_text: str
) -> list[Citation]:
    """Choose which stored notes to cite for an answer. Pure and unit-testable.

    Two structural gates, both required, so citation correctness never depends on
    matching refusal prose:

    1. Retrieved: the note's data_id appears in cognee's Evidence block (verbatim
       text is a fallback when the id is not printed). No Evidence -> nothing
       retrieved -> no citations.
    2. Used: the answer contains a distinctive term from the note, i.e. a
       significant word shared by note and answer that is NOT already in the
       query. A refusal only echoes the query subject, so it yields no
       distinctive term and cites nothing, however it is worded.

    Each retrieved data_id resolves back through the ingest-time record to its
    source message (transport + deeplink), so the citation points at the real
    Telegram / web message, not just the Evidence text.
    """
    if not evidence.strip():
        return []

    retrieved_ids = set(_EVIDENCE_DATA_ID_RE.findall(evidence))
    evidence_lower = evidence.lower()
    answer_terms = _significant_terms(answer_text)
    query_terms = _significant_terms(query_text)

    cited: list[_CitationRecord] = []
    seen: set[str] = set()
    for record in records:
        rid = str(record.data_id)
        if rid in seen:
            continue
        retrieved = rid in retrieved_ids or record.text.strip().lower() in evidence_lower
        if not retrieved:
            continue
        distinctive = (_significant_terms(record.text) & answer_terms) - query_terms
        if distinctive:
            cited.append(record)
            seen.add(rid)

    return [
        Citation(
            content=r.text,
            source_transport=r.transport,
            source_ref=r.deeplink or f"{r.transport}:{r.source}",
            timestamp=r.ts,
        )
        for r in cited
    ]


# Generic words that carry no distinguishing content (plus refusal boilerplate).
_STOPWORDS = frozenset(
    """
    a an and are as at be been being by do does did done for from had has have how
    i in into is it its me my no not of on or our that the their them then there
    these they this those to was were what when where which who whom why will with
    you your about above after again all also any because before below between both
    but can could would should over under out up down if so such than too very
    provided context information based mention determine contain contains details
    sorry apologize unfortunately unable relevant regarding specify specified
    found find available covered
    """.split()
)


def _normalize(token: str) -> str:
    """Crude stem so inflections collapse (open/opens/opened -> open).

    Needed because a refusal often restates the query using the note's
    inflection ("when the archive opens"); without collapsing, "opens" would
    look distinctive from the query's "open" and wrongly earn a citation.
    """
    for suffix in ("ing", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _significant_terms(text: str) -> set[str]:
    """Normalized content words of a string: lowercased, punctuation-stripped,
    stopwords out, lightly stemmed. Short-but-meaningful tokens like "5th"
    (len >= 2) are kept so date/number facts still count as distinctive.
    """
    terms = set()
    for raw in text.lower().split():
        token = raw.strip(".,;:!?\"'()[]{}<>")
        if len(token) >= 2 and token not in _STOPWORDS:
            terms.add(_normalize(token))
    return terms


def _render_result(item: object) -> str:
    for attr in ("text", "content", "answer"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


_NO_ANSWER_MARKERS = (
    "no information",
    "no relevant information",
    "not enough information",
    "no data about",
    "cannot answer",
    "could not find",
    "couldn't find",
    "do not have information",
    "don't have information",
)


def _is_no_answer(answer_text: str) -> bool:
    """Best-effort detection of a refusal, for display polish ONLY.

    Used only to render an already-uncited refusal as the friendly empty
    message. It is deliberately NOT the citation-correctness mechanism: prose
    matching an LLM refusal is fragile (phrasings like "the context does not
    mention ..." would slip past this list), so citation correctness lives in
    select_citations' structural check instead. A refusal this list misses just
    shows cognee's own wording, still with zero citations.
    """
    low = answer_text.lower()
    return any(marker in low for marker in _NO_ANSWER_MARKERS)


def _split_evidence(text: str) -> tuple[str, str]:
    """Split a recall result into (answer prose, Evidence block).

    cognee renders the grounded references under an ``Evidence:`` header. If
    there is no such header, cognee retrieved nothing to ground the answer, and
    the Evidence half is empty.
    """
    index = text.find(_EVIDENCE_HEADER)
    if index == -1:
        return text.strip(), ""
    return text[:index].strip(), text[index + len(_EVIDENCE_HEADER) :].strip()
