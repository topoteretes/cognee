from typing import Any, List, Optional, Union
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError

logger = get_logger("SkillsRetriever")

SKILL_COLLECTION = "Skill_search_text"

# Strict dataset/active filtering shrinks the candidate set, so fetch more
# than top_k from the vector engine and trim after filtering.
_FETCH_MULTIPLIER = 4
_MIN_FETCH_LIMIT = 20


def _project_skill_payload(payload: dict) -> dict:
    """Project a Skill vector payload onto the metadata-only skill shape.

    Mirrors ``list_skills._skill_to_dict`` but works on raw payload dicts and
    never includes ``procedure`` / ``skill_text`` / ``search_text``: search
    results must preserve the progressive-disclosure design — bodies load via
    the ``load_skill`` tool or ``GET /skills/{skill_id}``. ``.get()`` with
    defaults keeps skills ingested before newer fields existed from crashing.
    """
    return {
        "id": str(payload.get("id") or ""),
        "name": payload.get("name") or "",
        "description": payload.get("description") or "",
        "maintainer": payload.get("maintainer") or "",
        "maintainer_url": payload.get("maintainer_url") or "",
        "version": payload.get("skill_version") or "",
        "tags": list(payload.get("tags") or []),
        "license": payload.get("license") or "",
        "declared_tools": list(payload.get("declared_tools") or []),
        "dataset_scope": [str(entry) for entry in (payload.get("dataset_scope") or [])],
        "is_active": bool(payload.get("is_active", True)),
        "source_repo_url": payload.get("source_repo_url") or "",
        "source_dir": payload.get("source_dir") or "",
    }


class SkillsRetriever(BaseRetriever):
    """
    Retriever for semantic discovery of dataset-scoped Skill playbooks.

    Searches the ``Skill_search_text`` vector collection (populated by skill
    ingestion via the standard Embeddable indexing path) and returns
    metadata-only skill payloads — never the procedure body.

    Requires exactly one explicit dataset: only skills whose ``dataset_scope``
    contains that dataset (and that are ``is_active``) are returned. Skills
    with an empty ``dataset_scope`` are excluded.

    A missing collection returns an empty result instead of raising
    ``NoDataError`` (unlike SummariesRetriever): "no skills ingested yet" is a
    normal state, and the recall skill gate must degrade to a no-op.

    Public methods:
    - __init__
    - get_retrieved_objects
    - get_context_from_objects
    - get_completion_from_context
    """

    # Deterministic, non-generative search type: skip the conversational
    # session analysis (which may call an LLM before retrieval).
    supports_session_turn_preparation = False

    def __init__(
        self,
        top_k: Optional[int] = 5,
        dataset_id: Optional[Union[str, UUID]] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize retriever with search parameters. ``dataset_id`` is required."""
        if dataset_id is None:
            raise QueryValidationError(
                message="SKILLS search requires exactly one explicit dataset."
            )
        self.top_k = top_k if top_k is not None else 5
        self.dataset_id = str(dataset_id)
        self.session_id = session_id

    async def get_retrieved_objects(self, query: str) -> Any:
        """
        Retrieves skill hits for the query, filtered to this dataset.

        Over-fetches from the vector engine, then keeps only active skills
        whose ``dataset_scope`` contains the retriever's dataset, deduplicated
        by id and trimmed to ``top_k``.
        """
        logger.info(
            f"Starting skill retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        unified = await get_unified_engine()
        vector_engine = unified.vector

        fetch_limit = max(self.top_k * _FETCH_MULTIPLIER, _MIN_FETCH_LIMIT)

        try:
            results = await vector_engine.search(
                SKILL_COLLECTION, query, limit=fetch_limit, include_payload=True
            )
        except CollectionNotFoundError:
            logger.info(
                "%s collection not found — no skills ingested yet; returning no results",
                SKILL_COLLECTION,
            )
            return []

        filtered = []
        seen_ids: set = set()
        for result in results:
            payload = getattr(result, "payload", None) or {}
            if not payload.get("is_active", True):
                continue
            scope = [str(entry) for entry in (payload.get("dataset_scope") or [])]
            if self.dataset_id not in scope:
                continue
            payload_id = str(payload.get("id") or getattr(result, "id", "") or "")
            if payload_id:
                if payload_id in seen_ids:
                    continue
                seen_ids.add(payload_id)
            filtered.append(result)
            if len(filtered) >= self.top_k:
                break

        logger.info(f"Found {len(filtered)} in-scope skill(s) from vector search")
        return filtered

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Formats retrieved skills as a name + description listing.

        Same shape the agentic retriever puts in its system prompt, so the
        context is directly usable to offer skills to an LLM without leaking
        procedure bodies.
        """
        if not retrieved_objects:
            return ""
        lines = []
        for result in retrieved_objects:
            payload = getattr(result, "payload", None) or {}
            name = payload.get("name") or ""
            description = payload.get("description") or ""
            lines.append(f"- `{name}`: {description}")
        return "\n".join(lines)

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        """
        Returns metadata-only skill payloads; no LLM completion is generated.

        Each dict carries the projected skill fields plus the vector ``score``
        (raw backend distance — lower is better).
        """
        if not retrieved_objects:
            return []
        completions = []
        for result in retrieved_objects:
            payload = getattr(result, "payload", None) or {}
            projected = _project_skill_payload(payload)
            score = getattr(result, "score", None)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                projected["score"] = float(score)
            completions.append(projected)
        logger.info(f"Returning {len(completions)} skill payload(s)")
        return completions
