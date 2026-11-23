"""Recall's index mode and expand-by-id.

``result_mode="index"`` returns what an ``only_context`` hybrid recall retrieved as rows
instead of one rendered prompt: one memory row per passage, entity and fact (a short
``text``, its id, rank, score, version, conflicts and sources) and one frame row per part
of the prompt that has no id (template halves, session layer, grounding, global context).
Nothing is dropped: frame rows, expanded bodies and the conflict notes rebuild the
prompt regular mode returns. ``expand_items`` then returns the full bodies of the rows a
caller picked. Both render bodies with the retriever's own renderers, so a row's
``chars`` and ``version`` describe exactly the body expand returns.
"""

from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.models import Data
from cognee.modules.provenance.edge_evidence.lookup import get_edge_evidence_records
from cognee.modules.recall.methods.normalize_search_payload import search_result_kind
from cognee.modules.recall.types.RecallIndex import (
    ExpandItem,
    IndexRowMetadata,
    RecordSource,
)
from cognee.modules.recall.types.SearchResultItem import SearchResultItem
from cognee.modules.retrieval.hybrid.context import attached_conflict_texts, render_passage
from cognee.modules.retrieval.hybrid.entities import render_entity
from cognee.modules.retrieval.utils.results import display_value, payload, result_id
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType
from cognee.modules.users.exceptions import PermissionDeniedError

# Listed by /health (``get_capabilities``): this server answers result_mode="index" and
# expand_items.
RECALL_INDEX_CAPABILITY = "recall_index_v1"
# The issue's preview length (SDK-1099): a row's ``text`` is its title or first line,
# cut to this many characters.
INDEX_TEXT_CHARS = 200
# The issue's expand bound (SDK-1099): records per expand call.
MAX_EXPAND_ITEMS = 20


def record_version(body: str) -> str:
    """A record's version: the content hash of the body expand returns for it."""
    return compute_content_hash(body)


async def build_index_items(payload_: SearchResultPayload) -> list[SearchResultItem]:
    """One dataset's ``only_context`` hybrid result as index rows, in prompt order.

    Raises ``ValueError`` when the payload carries no prompt frame (retrieval found
    nothing returns [] instead) or its template cannot be split around the context.
    """
    retrieved = payload_.result_object
    if not payload_.user_prompt:
        return []
    frame = payload_.context_frame
    if not isinstance(retrieved, dict) or frame is None:
        raise ValueError("The index needs a hybrid only_context result with its prompt frame.")
    if frame["template_prefix"] is None:
        raise ValueError("The retriever's user template cannot be split around the context.")

    chunk_scores = retrieved.get("chunk_scores") or {}
    chunks = retrieved.get("chunks") or []
    entities = retrieved.get("entities") or []
    conflict_texts = attached_conflict_texts(retrieved)
    facts = [
        fact for fact in retrieved.get("facts") or [] if fact.get("text") not in conflict_texts
    ]

    created_at = await _created_at_by_data_id(
        [display_value(payload(chunk).get("document_id")) for chunk in chunks]
    )
    entity_sources = await _entity_sources(entities, payload_.dataset_id)

    rows: list[SearchResultItem] = []

    def frame_row(record_kind: str, text: str) -> None:
        if text:
            rows.append(
                _row(payload_, text, IndexRowMetadata(record_kind=record_kind, chars=len(text)))
            )

    frame_row("session_history", frame["history"])
    frame_row("template_prefix", frame["template_prefix"])
    frame_row("grounding", frame["grounding"])
    frame_row("global_context", frame["global_context"])

    for rank, chunk in enumerate(chunks, start=1):
        body = render_passage(chunk)
        if not body:
            continue
        chunk_id = result_id(chunk)
        scores = chunk_scores.get(chunk_id, {})
        rows.append(
            _memory_row(
                payload_,
                record_kind="passage",
                record_id=chunk_id,
                rank=rank,
                preview=_first_line(display_value(payload(chunk).get("text")) or body),
                body=body,
                score=scores.get("score"),
                distance=scores.get("distance"),
                conflicts=list(payload(chunk).get("conflicts", [])),
                sources=[_passage_source(chunk, created_at)],
            )
        )

    for rank, entity in enumerate(entities, start=1):
        body = render_entity(entity)
        if not body:
            continue
        rows.append(
            _memory_row(
                payload_,
                record_kind="entity",
                record_id=entity["id"],
                rank=rank,
                preview=_first_line(body).removeprefix("### "),
                body=body,
                score=entity.get("score"),
                conflicts=list(entity.get("conflicts", [])),
                sources=entity_sources.get(entity["id"], []),
            )
        )

    for rank, fact in enumerate(facts, start=1):
        rows.append(
            _memory_row(
                payload_,
                record_kind="fact",
                record_id=fact["id"],
                rank=rank,
                preview=fact["text"],
                body=fact["text"],
                score=fact.get("score"),
                conflicts=[],
                sources=None,
            )
        )

    frame_row("template_suffix", frame["template_suffix"])
    frame_row("session_guidance", frame["guidance"])
    return rows


async def expand_records(
    items: list[ExpandItem],
    *,
    query_text: str,
    user: Any,
    retriever_kwargs: dict,
) -> list[SearchResultItem]:
    """Full bodies for index rows, one result per requested item, in request order.

    Items are grouped by dataset: read access is checked once per dataset and its stores
    are opened once. A record the dataset no longer holds is ``missing``; a dataset the
    caller cannot read makes its records ``forbidden``. ``retriever_kwargs`` must be the
    index call's settings (top_k, node_name, retriever_specific_config, ...), so entity
    bullets come out as the index call ranked them.
    """
    unique_items = list(
        dict.fromkeys((item.dataset_id, item.record_kind, item.record_id) for item in items)
    )
    by_dataset: dict[UUID, list[tuple[str, str]]] = defaultdict(list)
    for dataset_id, record_kind, record_id in unique_items:
        by_dataset[dataset_id].append((record_kind, record_id))

    results: dict[tuple[UUID, str, str], SearchResultItem] = {}
    for dataset_id, records in by_dataset.items():
        dataset = await _readable_dataset(dataset_id, user)
        if dataset is None:
            for record_kind, record_id in records:
                results[(dataset_id, record_kind, record_id)] = _status_row(
                    dataset_id, None, record_kind, record_id, "forbidden"
                )
            continue
        ids_by_kind: dict[str, list[str]] = defaultdict(list)
        for record_kind, record_id in records:
            ids_by_kind[record_kind].append(record_id)
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            retriever = await get_search_type_retriever_instance(
                SearchType.HYBRID_COMPLETION, query_text, dataset=dataset, **retriever_kwargs
            )
            found = await retriever.expand_records(query_text, dict(ids_by_kind))
        results.update(await _expanded_rows(dataset_id, dataset.name, records, found))
    return [results[key] for key in unique_items]


async def _readable_dataset(dataset_id: UUID, user: Any):
    """The dataset when ``user`` may read it, else None (unknown and unreadable look alike)."""
    try:
        datasets = await get_authorized_existing_datasets([dataset_id], "read", user)
    except (PermissionDeniedError, DatasetNotFoundError):
        return None
    return datasets[0] if datasets else None


async def _expanded_rows(
    dataset_id: UUID, dataset_name: str | None, records: list[tuple[str, str]], found: dict
) -> dict:
    chunks = [found[key]["item"] for key in found if key[0] == "passage"]
    entities = [found[key]["item"] for key in found if key[0] == "entity"]
    created_at = await _created_at_by_data_id(
        [display_value(payload(chunk).get("document_id")) for chunk in chunks]
    )
    entity_sources = await _entity_sources(entities, dataset_id)

    rows = {}
    for record_kind, record_id in records:
        record = found.get((record_kind, record_id))
        if record is None or not record["body"]:
            rows[(dataset_id, record_kind, record_id)] = _status_row(
                dataset_id, dataset_name, record_kind, record_id, "missing"
            )
            continue
        body = record["body"]
        if record_kind == "passage":
            sources = [_passage_source(record["item"], created_at)]
        elif record_kind == "entity":
            sources = entity_sources.get(record_id, [])
        else:
            sources = None
        metadata = IndexRowMetadata(
            record_kind=record_kind,
            record_id=record_id,
            chars=len(body),
            version=record_version(body),
            conflicts=record["conflicts"],
            sources=sources,
            status="available",
        )
        rows[(dataset_id, record_kind, record_id)] = SearchResultItem(
            kind=search_result_kind(SearchType.HYBRID_COMPLETION),
            search_type=SearchType.HYBRID_COMPLETION,
            text=body,
            dataset_id=str(dataset_id),
            dataset_name=dataset_name,
            metadata=metadata.model_dump(mode="json"),
        )
    return rows


def _status_row(
    dataset_id: UUID, dataset_name: str | None, record_kind: str, record_id: str, status: str
) -> SearchResultItem:
    metadata = IndexRowMetadata(
        record_kind=record_kind, record_id=record_id, chars=0, status=status
    )
    return SearchResultItem(
        kind=search_result_kind(SearchType.HYBRID_COMPLETION),
        search_type=SearchType.HYBRID_COMPLETION,
        text="",
        dataset_id=str(dataset_id),
        dataset_name=dataset_name,
        metadata=metadata.model_dump(mode="json"),
    )


def _memory_row(
    payload_: SearchResultPayload,
    *,
    record_kind: str,
    record_id: str | None,
    rank: int,
    preview: str,
    body: str,
    score: float | None,
    conflicts: list[str],
    sources: list[RecordSource] | None,
    distance: float | None = None,
) -> SearchResultItem:
    metadata = IndexRowMetadata(
        record_kind=record_kind,
        record_id=record_id,
        rank=rank,
        score=score,
        distance=distance,
        chars=len(body),
        version=record_version(body),
        conflicts=conflicts,
        sources=sources,
    )
    return _row(payload_, preview[:INDEX_TEXT_CHARS], metadata, score=score)


def _row(
    payload_: SearchResultPayload,
    text: str,
    metadata: IndexRowMetadata,
    *,
    score: float | None = None,
) -> SearchResultItem:
    return SearchResultItem(
        kind=search_result_kind(payload_.search_type),
        search_type=payload_.search_type,
        text=text,
        score=score,
        dataset_id=str(payload_.dataset_id) if payload_.dataset_id else None,
        dataset_name=payload_.dataset_name,
        metadata=metadata.model_dump(mode="json"),
    )


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _passage_source(chunk: Any, created_at: dict[str, str]) -> RecordSource:
    chunk_payload = payload(chunk)
    data_id = display_value(chunk_payload.get("document_id")) or ""
    return RecordSource(
        data_id=data_id,
        document_name=display_value(chunk_payload.get("document_name")),
        chunk_id=result_id(chunk),
        created_at=created_at.get(data_id),
    )


async def _entity_sources(entities: list[dict], dataset_id: Any) -> dict[str, list[RecordSource]]:
    """The documents behind each entity's shown bullets, from the edge-evidence table.

    Only edges cognify extracted from document chunks have evidence; an entity known
    only through improve(), session bridging or the code graph has none.
    """
    if not entities or not dataset_id:
        return {}
    edge_owner: dict[UUID, str] = {}
    for entity in entities:
        for edge in entity.get("edges", []):
            edge_object_id = display_value(edge.get("edge_object_id"))
            if edge_object_id:
                edge_owner.setdefault(UUID(edge_object_id), entity["id"])
    if not edge_owner:
        return {}
    records = await get_edge_evidence_records(list(edge_owner), UUID(str(dataset_id)))
    created_at = await _created_at_by_data_id([str(record.data_id) for record in records])
    sources: dict[str, dict[tuple[str, str], RecordSource]] = defaultdict(dict)
    for record in records:
        key = (str(record.data_id), str(record.chunk_id))
        sources[edge_owner[record.edge_id]].setdefault(
            key,
            RecordSource(
                data_id=key[0],
                document_name=record.document_name,
                chunk_id=key[1],
                created_at=created_at.get(key[0]),
            ),
        )
    return {entity_id: list(by_key.values()) for entity_id, by_key in sources.items()}


async def _created_at_by_data_id(data_ids: list[str | None]) -> dict[str, str]:
    ids = {UUID(data_id) for data_id in data_ids if data_id}
    if not ids:
        return {}
    async with get_relational_engine().get_async_session() as session:
        rows = (
            await session.execute(select(Data.id, Data.created_at).where(Data.id.in_(ids)))
        ).all()
    return {str(data_id): created.isoformat() for data_id, created in rows if created is not None}
