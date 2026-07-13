"""Deterministic, LLM-free helpers for building reference (Evidence) blocks.

Evidence is grounded in the generated answer, not in whatever happened to be
retrieved before the LLM was called:

- ``format_chunk_references`` builds an Evidence block from retrieved vector
  payloads, keeping only chunks that share significant terms with the answer
  (the ``RAG_COMPLETION`` / chunk path, where candidates are the chunks the
  LLM actually read).
- ``build_answer_grounded_chunk_references`` runs the answer text as a vector
  query against the chunk index and formats the results (the graph completion
  path, where the LLM context is not chunk-shaped).
- ``append_chunk_evidence`` / ``append_answer_grounded_evidence`` apply the
  above to a list of completions, one Evidence block per string completion.

All helpers are pure with respect to the LLM (no model calls) so they can be
unit tested in isolation. All return ``""`` (or the completions unchanged)
when there is nothing usable, and never raise on backend failures.
"""

import re
from typing import Any, List, Optional, Set, Tuple

from cognee.shared.logging_utils import get_logger

logger = get_logger("references")

# Header emitted on its own line above the bullets. Kept here so both helpers
# and the wiring code agree on the exact literal.
EVIDENCE_HEADER = "Evidence:"

# Maximum length of a rendered text snippet (characters) before truncation.
_SNIPPET_MAX_CHARS = 160

# Hard upper bound on bullets regardless of the requested limit (3-5 range).
_MAX_BULLETS = 5
_MIN_LIMIT = 3

# Vector collection holding document chunks (same one ChunksRetriever queries).
_CHUNK_COLLECTION = "DocumentChunk_text"

# How many vector candidates to fetch before answer-overlap filtering.
_CANDIDATE_POOL = 10

# Common English words excluded from answer/chunk term overlap scoring.
_STOPWORDS = frozenset(
    """
    a about above after again all also an and any are as at be because been
    before being below between both but by can did do does doing down during
    each few for from further had has have having he her here hers him his how
    i if in into is it its just me more most my no nor not of off on once only
    or other our ours out over own same she should so some such than that the
    their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will
    with you your yours
    """.split()
)


def _clamp_limit(limit: int) -> int:
    """Clamp the requested bullet limit into the contracted 3-5 range."""
    if limit < _MIN_LIMIT:
        return _MIN_LIMIT
    if limit > _MAX_BULLETS:
        return _MAX_BULLETS
    return limit


def _clean_str(value: Any) -> Optional[str]:
    """Return a stripped string, or None if the value is unusable.

    Missing, null, non-string, or empty/whitespace-only values are treated as
    unusable (the common state for data indexed before reference fields
    existed).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        # Numbers etc. are not valid document names / text; reject defensively.
        return None
    stripped = value.strip()
    return stripped or None


def _snippet(text: str) -> str:
    """Collapse whitespace and truncate text into a short snippet."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_MAX_CHARS:
        return collapsed
    return collapsed[: _SNIPPET_MAX_CHARS - 1].rstrip() + "…"


def _chunk_number(payload: dict) -> Optional[int]:
    """Resolve the 1-based display number from payload.

    Prefers an explicit ``chunk_number`` if present; otherwise derives it from
    the 0-based ``chunk_index`` as ``chunk_index + 1``. Returns None when no
    usable index information is present.
    """
    chunk_number = payload.get("chunk_number")
    if isinstance(chunk_number, bool):  # guard: bool is an int subclass
        chunk_number = None
    if isinstance(chunk_number, int) and chunk_number > 0:
        return chunk_number

    chunk_index = payload.get("chunk_index")
    if isinstance(chunk_index, bool):
        chunk_index = None
    if isinstance(chunk_index, int) and chunk_index >= 0:
        return chunk_index + 1

    return None


def _get_payload(obj: Any) -> Optional[dict]:
    """Extract a payload dict from a retrieved object.

    Retrieved objects are ``ScoredResult`` instances exposing ``.payload`` as a
    dict, but we also tolerate a raw dict or any object carrying a ``payload``
    attribute so the helper stays unit-testable without constructing a full
    ``ScoredResult``.
    """
    if isinstance(obj, dict):
        # Either the object IS the payload, or it wraps one under "payload".
        inner = obj.get("payload")
        if isinstance(inner, dict):
            return inner
        return obj

    payload = getattr(obj, "payload", None)
    if isinstance(payload, dict):
        return payload
    return None


def _provenance_suffix(data_id: Optional[str], chunk_id: Optional[str]) -> str:
    """Render a '(data_id: …, chunk_id: …)' annotation for whichever ids exist.

    Lets a reader map the citation back to the ingested data item and the exact
    cited chunk, instead of only a (possibly auto-generated) document name and a
    positional chunk number.
    """
    parts = []
    if data_id:
        parts.append(f"data_id: {data_id}")
    if chunk_id:
        parts.append(f"chunk_id: {chunk_id}")
    return f" ({', '.join(parts)})" if parts else ""


def _chunk_id(obj: Any, payload: dict) -> Optional[str]:
    """Resolve a stable chunk id for dedup, preferring the object id."""
    obj_id = getattr(obj, "id", None)
    if obj_id is not None:
        return str(obj_id)
    payload_id = payload.get("id")
    if payload_id is not None:
        return str(payload_id)
    # No stable id: fall back to (document_name, chunk_number) signature so we
    # still avoid duplicate bullets, computed by the caller from the payload.
    return None


def _significant_terms(text: str) -> Set[str]:
    """Lowercased alphanumeric terms of an answer, minus stopwords and stubs."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) >= 3 and token not in _STOPWORDS}


def format_chunk_references(
    retrieved_objects: Any, answer: Optional[str] = None, limit: int = 5
) -> str:
    """Build an Evidence block from retrieved vector payloads, grounded in the answer.

    Reads ``payload["document_name"]``, ``payload["chunk_number"]`` (falling back
    to ``payload["chunk_index"] + 1``), and ``payload["text"]`` from each
    retrieved object. Entries missing usable document name or chunk-number
    metadata are skipped. Results are deduplicated by chunk id and capped at
    3-5 bullets.

    When ``answer`` is provided, candidates that share no significant terms
    with the answer are dropped and the remainder is ranked by term overlap,
    so bullets reflect answer provenance rather than retrieval order. An
    answer with no significant terms cannot be grounded and yields ``""``.

    Parameters
    ----------
    retrieved_objects:
        An iterable of retrieved vector results (``ScoredResult``-like objects
        exposing a ``.payload`` dict), or raw payload dicts.
    answer:
        The generated answer text used to filter and rank candidates. When
        None, candidates keep their retrieval order unfiltered.
    limit:
        Desired maximum number of bullets, clamped into the 3-5 range.

    Returns
    -------
    str
        A multi-line Evidence block prefixed by an ``Evidence:`` header, or an
        empty string when nothing usable was found.
    """
    if not retrieved_objects:
        return ""

    try:
        iterator = list(retrieved_objects)
    except TypeError:
        return ""

    answer_terms: Optional[Set[str]] = None
    if answer is not None:
        answer_terms = _significant_terms(answer)
        if not answer_terms:
            # Nothing to ground the citation in (e.g. "Yes."): omit Evidence
            # rather than presenting unverifiable retrieval order as provenance.
            return ""

    # (overlap_score, document_name, number, text, data_id, chunk_id) per candidate.
    candidates: List[Tuple[int, str, int, str, Optional[str], Optional[str]]] = []
    seen: set = set()

    for obj in iterator:
        payload = _get_payload(obj)
        if payload is None:
            continue

        document_name = _clean_str(payload.get("document_name"))
        number = _chunk_number(payload)
        text = _clean_str(payload.get("text"))

        # Document name and a chunk number are both required to ground the
        # citation; text is required for a meaningful snippet.
        if document_name is None or number is None or text is None:
            continue

        chunk_id = _chunk_id(obj, payload)
        # document_id == the ingested Data item's id (cognify sets
        # Document.id = data.id), i.e. the dataId a caller needs to map a
        # citation back to the document they ingested.
        data_id = _clean_str(payload.get("document_id"))

        dedup_key = chunk_id or f"{document_name}#{number}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        score = 0
        if answer_terms is not None:
            chunk_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
            score = len(answer_terms & chunk_terms)
            if score == 0:
                # No term from the answer appears in this chunk: it is almost
                # certainly not a source of the answer.
                continue

        candidates.append((score, document_name, number, text, data_id, chunk_id))

    if not candidates:
        return ""

    if answer_terms is not None:
        # Stable sort: highest answer overlap first, retrieval order as tiebreak.
        candidates.sort(key=lambda candidate: -candidate[0])

    max_bullets = _clamp_limit(limit)
    bullets = [
        f"- chunk {number} of document {document_name}"
        f'{_provenance_suffix(data_id, chunk_id)}: "{_snippet(text)}"'
        for _, document_name, number, text, data_id, chunk_id in candidates[:max_bullets]
    ]

    return EVIDENCE_HEADER + "\n" + "\n".join(bullets)


async def build_answer_grounded_chunk_references(
    answer: str, vector_engine: Any, limit: int = 5
) -> str:
    """Build an Evidence block by running the answer as a vector query over chunks.

    This grounds Evidence in the answer text itself, independent of how the
    original retrieval was done (graph traversal, triplets, ...): the answer is
    embedded once and matched against the existing chunk index, then candidates
    are additionally filtered by term overlap with the answer.

    Never raises: a missing collection or backend failure degrades to ``""``.

    Parameters
    ----------
    answer:
        The generated answer text to ground.
    vector_engine:
        A vector engine exposing ``search(collection, query, limit, include_payload)``.
    limit:
        Desired maximum number of bullets, clamped into the 3-5 range.

    Returns
    -------
    str
        A multi-line Evidence block prefixed by an ``Evidence:`` header, or an
        empty string when nothing usable was found or the search failed.
    """
    cleaned_answer = _clean_str(answer)
    if cleaned_answer is None or vector_engine is None:
        return ""

    try:
        found_chunks = await vector_engine.search(
            _CHUNK_COLLECTION,
            cleaned_answer,
            limit=_CANDIDATE_POOL,
            include_payload=True,
        )
    except Exception as error:
        logger.debug(f"Answer-grounded chunk search failed: {error}")
        return ""

    return format_chunk_references(found_chunks, answer=cleaned_answer, limit=limit)


def append_chunk_evidence(
    completions: List[Any], retrieved_objects: Any, enabled: bool
) -> List[Any]:
    """Append an answer-grounded chunk Evidence block to string completions.

    Each string completion gets its own Evidence block, filtered and ranked by
    that completion's text against the retrieved candidates. Non-string
    completions (structured response models) are never touched, and an empty
    Evidence block leaves the completion unchanged.
    """
    if not enabled:
        return completions

    appended: List[Any] = []
    for completion in completions:
        if not isinstance(completion, str):
            appended.append(completion)
            continue
        evidence = format_chunk_references(retrieved_objects, answer=completion)
        appended.append(f"{completion}\n\n{evidence}" if evidence else completion)
    return appended


async def append_answer_grounded_evidence(completions: List[Any], enabled: bool) -> List[Any]:
    """Append an answer-grounded Evidence block to string completions.

    Each string completion is run as a vector query against the chunk index
    (see :func:`build_answer_grounded_chunk_references`). Non-string completions
    are never touched; any backend failure degrades to no Evidence.
    """
    if not enabled:
        return completions

    try:
        from cognee.infrastructure.databases.vector import get_vector_engine_async

        vector_engine = await get_vector_engine_async()
    except Exception as error:
        logger.debug(f"Unable to obtain vector engine for references: {error}")
        return completions

    appended: List[Any] = []
    for completion in completions:
        if not isinstance(completion, str):
            appended.append(completion)
            continue
        evidence = await build_answer_grounded_chunk_references(completion, vector_engine)
        appended.append(f"{completion}\n\n{evidence}" if evidence else completion)
    return appended
