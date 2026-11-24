"""Recall's cited mode and expand-by-id.

``result_mode="cited"`` returns the ``only_context`` prompt a hybrid recall builds, with a
tag line, ``[mem:<kind>/<dataset_id>/<record_id>@<version>]``, directly before every
passage, entity and fact, and one ``CitedRecord`` per tag in ``metadata["records"]`` (rank,
score, length, version, conflicts, sources), so a model can name the memory it used and a
caller can select records by score. ``expand_items`` reads records by id later, from a tag
or its fields. Both render bodies with the retriever's own renderers, so a record's
``chars`` and ``version`` describe exactly the body expand returns.
"""

import re
from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.models import Data
from cognee.modules.provenance.edge_evidence.lookup import get_edge_evidence_records
from cognee.modules.recall.methods.normalize_search_payload import (
    normalize_search_payload,
    search_result_kind,
)
from cognee.modules.recall.types.RecallCited import CitedRecord, ExpandItem, RecordSource
from cognee.modules.recall.types.SearchResultItem import SearchResultItem
from cognee.modules.retrieval.hybrid.context import attached_conflict_texts, render_passage
from cognee.modules.retrieval.hybrid.entities import render_entity
from cognee.modules.retrieval.utils.conflict_context import format_conflicts
from cognee.modules.retrieval.utils.results import display_value, payload, result_id
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType
from cognee.modules.users.exceptions import PermissionDeniedError

# Listed by /health (``get_capabilities``): this server answers result_mode="cited" and
# expand_items.
RECALL_CITED_CAPABILITY = "recall_cited_v1"
# The issue's expand bound (SDK-1099): records per expand call.
MAX_EXPAND_ITEMS = 20

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_TAG = re.compile(
    rf"\[mem:(?P<kind>passage|entity|fact)/(?P<dataset>{_UUID})/(?P<record>{_UUID})"
    r"@(?P<version>[0-9a-f]{32})\]"
)


def record_version(body: str) -> str:
    """A record's version: the content hash of the body expand returns for it."""
    return compute_content_hash(body)


def record_tag(record_kind: str, dataset_id: Any, record_id: Any, version: str) -> str:
    """The tag a cited prompt puts before a record: kind, dataset, id and version."""
    return f"[mem:{record_kind}/{dataset_id}/{record_id}@{version}]"


def parse_record_tag(tag: str) -> ExpandItem:
    """The record a tag names. Raises ``ValueError`` for anything but one whole tag."""
    match = _TAG.fullmatch(tag.strip())
    if match is None:
        raise ValueError(
            f"not a memory tag: {tag!r}; expected "
            "[mem:<passage|entity|fact>/<dataset_id>/<record_id>@<version>]"
        )
    return ExpandItem(
        dataset_id=match["dataset"], record_kind=match["kind"], record_id=match["record"]
    )


async def build_cited_item(payload_: SearchResultPayload) -> list[SearchResultItem]:
    """One dataset's ``only_context`` hybrid result as its cited prompt, or [] when
    retrieval found nothing.

    The prompt is assembled as ``format_hybrid_context`` and ``build_completion_prompts``
    assemble regular mode's (the frame comes from ``payload_.context_frame``), with each
    record's tag line before its body. Raises ``CitedRequiresHybridError`` when hybrid
    handed this dataset to another search type (``hybrid_deferral_reason``: the dataset
    has no chunk collection), and ``CitedTemplateNotSplittableError`` when the retriever's
    user template cannot be split around the context.
    """
    retrieved = payload_.result_object
    if not payload_.user_prompt:
        return []
    if payload_.search_type != SearchType.HYBRID_COMPLETION:
        raise CogneeValidationError(
            message=f"result_mode='cited' needs hybrid retrieval, but dataset "
            f"'{payload_.dataset_name}' was answered by {payload_.search_type}: it has no "
            "passages to cite (no DocumentChunk_text or DltRow_text collection). Leave it "
            "out of datasets, or use result_mode='full'.",
            name="CitedRequiresHybridError",
        )
    frame = payload_.context_frame
    if not isinstance(retrieved, dict) or frame is None:
        # Request-level deferrals are refused before the search runs and the dataset-level
        # one above, so a hybrid only_context payload without its frame is a bug.
        raise CogneeSystemError(
            message="A hybrid only_context result reached the cited recall without its "
            "prompt frame.",
            name="CitedFrameMissingError",
        )
    if frame["template_prefix"] is None:
        raise CogneeValidationError(
            message="result_mode='cited' needs a user prompt template that renders the "
            "context exactly once; this retriever's template cannot be split around it.",
            name="CitedTemplateNotSplittableError",
        )

    dataset_id = str(payload_.dataset_id)
    chunk_scores = retrieved.get("chunk_scores") or {}
    chunks = retrieved.get("chunks") or []
    entities = retrieved.get("entities") or []
    conflict_texts = attached_conflict_texts(retrieved)
    facts = [
        fact
        for fact in retrieved.get("facts") or []
        if fact.get("text") and fact["text"] not in conflict_texts
    ]
    created_at = await _created_at_by_data_id(
        [display_value(payload(chunk).get("document_id")) for chunk in chunks]
    )
    entity_sources = await _entity_sources(entities, payload_.dataset_id)

    records: list[CitedRecord] = []

    def cite(record_kind: str, record_id: str, body: str, **fields) -> str:
        """``body`` behind its tag line, recorded in ``records``."""
        version = record_version(body)
        tag = record_tag(record_kind, dataset_id, record_id, version)
        records.append(
            CitedRecord(
                tag=tag,
                record_kind=record_kind,
                dataset_id=dataset_id,
                record_id=record_id,
                version=version,
                chars=len(body),
                **fields,
            )
        )
        return f"{tag}\n{body}"

    passages = []
    for rank, chunk in enumerate(chunks, start=1):
        body = render_passage(chunk)
        if not body:
            continue
        chunk_id = result_id(chunk)
        scores = chunk_scores.get(chunk_id, {})
        passages.append(
            cite(
                "passage",
                chunk_id,
                body,
                rank=rank,
                score=scores.get("score"),
                distance=scores.get("distance"),
                conflicts=list(payload(chunk).get("conflicts", [])),
                sources=[_passage_source(chunk, created_at)],
            )
        )

    entity_blocks = []
    for rank, entity in enumerate(entities, start=1):
        body = render_entity(entity)
        if not body:
            continue
        entity_blocks.append(
            cite(
                "entity",
                entity["id"],
                body,
                rank=rank,
                score=entity.get("score"),
                conflicts=list(entity.get("conflicts", [])),
                sources=entity_sources.get(entity["id"], []),
            )
        )

    fact_lines = [
        cite("fact", fact["id"], f"- {fact['text']}", rank=rank, score=fact.get("score"))
        for rank, fact in enumerate(facts, start=1)
    ]

    # The sections and joiners of ``format_hybrid_context``.
    context = "\n\n".join(
        section
        for section in (
            frame["global_context"],
            _section("## Relevant passages\n", "\n---\n", passages),
            _section("## Relevant entities\n", "\n\n", entity_blocks),
            _section("## Related facts\n", "\n", fact_lines),
            format_conflicts(conflict_texts),
        )
        if section
    )
    if frame["grounding"]:
        context = f"{frame['grounding']}\n\n{context}"
    # The session layer around the rendered template, as ``build_completion_prompts``.
    rendered = frame["template_prefix"] + context + frame["template_suffix"]
    text = "\n\n".join(part for part in (frame["history"], rendered, frame["guidance"]) if part)

    (item,) = normalize_search_payload(payload_)
    return [
        item.model_copy(
            update={
                "text": text,
                "raw": {"value": text},
                "metadata": {
                    **item.metadata,
                    "records": [record.model_dump(mode="json") for record in records],
                },
            }
        )
    ]


def _section(header: str, joiner: str, blocks: list[str]) -> str:
    return header + joiner.join(blocks) if blocks else ""


async def expand_records(
    items: list[ExpandItem],
    *,
    query_text: str,
    user: Any,
    retriever_kwargs: dict,
) -> list[SearchResultItem]:
    """Full bodies for cited records, one result per requested item, in request order.

    Items are grouped by dataset: read access is checked once per dataset and its stores
    are opened once. A record the dataset no longer holds is ``missing``; a dataset the
    caller cannot read makes its records ``forbidden``. ``retriever_kwargs`` must be the
    cited call's settings (top_k, node_name, retriever_specific_config, ...), so entity
    bullets come out as the cited call ranked them.
    """
    unique_items = list(
        dict.fromkeys((item.dataset_id, item.record_kind, str(item.record_id)) for item in items)
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
        # A fact's cited body is its "- " line in Related facts.
        body = f"- {record['body']}" if record_kind == "fact" else record["body"]
        if record_kind == "passage":
            sources = [_passage_source(record["item"], created_at)]
        elif record_kind == "entity":
            sources = entity_sources.get(record_id, [])
        else:
            sources = None
        version = record_version(body)
        metadata = CitedRecord(
            tag=record_tag(record_kind, dataset_id, record_id, version),
            record_kind=record_kind,
            dataset_id=str(dataset_id),
            record_id=record_id,
            version=version,
            chars=len(body),
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
    metadata = CitedRecord(
        tag=f"[mem:{record_kind}/{dataset_id}/{record_id}]",
        record_kind=record_kind,
        dataset_id=str(dataset_id),
        record_id=record_id,
        chars=0,
        status=status,
    )
    return SearchResultItem(
        kind=search_result_kind(SearchType.HYBRID_COMPLETION),
        search_type=SearchType.HYBRID_COMPLETION,
        text="",
        dataset_id=str(dataset_id),
        dataset_name=dataset_name,
        metadata=metadata.model_dump(mode="json"),
    )


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
    # An edge between two shown entities is a bullet of both, so its evidence is a
    # source of both.
    edge_entities: dict[UUID, list[str]] = defaultdict(list)
    for entity in entities:
        for edge in entity.get("edges", []):
            edge_object_id = display_value(edge.get("edge_object_id"))
            if edge_object_id and entity["id"] not in edge_entities[UUID(edge_object_id)]:
                edge_entities[UUID(edge_object_id)].append(entity["id"])
    if not edge_entities:
        return {}
    records = await get_edge_evidence_records(list(edge_entities), UUID(str(dataset_id)))
    created_at = await _created_at_by_data_id([str(record.data_id) for record in records])
    sources: dict[str, dict[tuple[str, str], RecordSource]] = defaultdict(dict)
    for record in records:
        key = (str(record.data_id), str(record.chunk_id))
        for entity_id in edge_entities[record.edge_id]:
            sources[entity_id].setdefault(
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
