"""Recall's cited mode and expand-by-id.

``result_mode="cited"`` returns the ``only_context`` prompt a hybrid recall builds, with a
tag line, ``[mem:<kind>/<dataset_id>/<record_id>@<version>]``, directly before every
passage, entity and fact, and one ``CitedRecord`` per tag in ``metadata["records"]`` (rank,
score, length, version, conflicts, sources), so a model can name the memory it used and a
caller can select records by score. ``expand_items`` reads records by id later, from a tag
or its fields, with no query. Both render bodies with the retriever's own renderers, so a
record's ``version`` describes exactly the body expand returns: for a passage or fact the
body the prompt shows, for an entity every one-hop edge (the prompt shows the ones the
query ranked first).
"""

import re
from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
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
from cognee.modules.retrieval.hybrid.entities import render_entity, render_full_entity
from cognee.modules.retrieval.utils.conflict_context import format_conflicts
from cognee.modules.retrieval.utils.results import display_value, payload, result_id
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_cited")

# Listed by /health (``get_capabilities``): this server answers result_mode="cited" and
# expand_items.
RECALL_CITED_CAPABILITY = "recall_cited_v1"
# The issue's expand bound (SDK-1099): records per expand call.
MAX_EXPAND_ITEMS = 20

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
# The dataset is left out for a record of the shared store (access control off).
_TAG = re.compile(
    rf"\[mem:(?P<kind>passage|entity|fact)/(?:(?P<dataset>{_UUID})/)?(?P<record>{_UUID})"
    r"@(?P<version>[0-9a-f]{32})\]"
)


def record_version(body: str) -> str:
    """A record's version: the content hash of the body expand returns for it."""
    return compute_content_hash(body)


def record_tag(record_kind: str, dataset_id: Any, record_id: Any, version: str) -> str:
    """The tag a cited prompt puts before a record: kind, dataset (when there is one), id
    and version."""
    dataset = f"{dataset_id}/" if dataset_id is not None else ""
    return f"[mem:{record_kind}/{dataset}{record_id}@{version}]"


def parse_record_tag(tag: str) -> ExpandItem:
    """The record a tag names. Raises ``ValueError`` for anything but one whole tag."""
    match = _TAG.fullmatch(tag.strip())
    if match is None:
        raise ValueError(
            f"not a memory tag: {tag!r}; expected "
            "[mem:<passage|entity|fact>/[<dataset_id>/]<record_id>@<version>]"
        )
    return ExpandItem(
        dataset_id=match["dataset"],
        record_kind=match["kind"],
        record_id=match["record"],
        version=match["version"],
    )


async def build_cited_items(
    payloads: list[SearchResultPayload], user: Any
) -> list[SearchResultItem]:
    """The cited prompt of every searched dataset hybrid answered, in payload order.

    A dataset hybrid handed to another search type (``hybrid_deferral_reason``: it has no
    chunk collection, a code graph for example) has no passages, entities or facts to
    tag, so it is left out, as one dataset without searchable memory never fails an
    unscoped search. Raises ``CitedRequiresHybridError`` only when every dataset that
    built a prompt was handed off.
    """
    prompted = [payload_ for payload_ in payloads if payload_.user_prompt]
    handed_off = [
        payload_ for payload_ in prompted if payload_.search_type != SearchType.HYBRID_COMPLETION
    ]
    if prompted and len(handed_off) == len(prompted):
        names = ", ".join(f"'{payload_.dataset_name}'" for payload_ in handed_off)
        raise CogneeValidationError(
            message=f"result_mode='cited' needs hybrid retrieval, but no searched dataset "
            f"has passages to cite (no DocumentChunk_text or DltRow_text collection): {names} "
            "answered with another search type. Search a dataset with documents, or use "
            "result_mode='full'.",
            name="CitedRequiresHybridError",
        )
    for payload_ in handed_off:
        logger.info(
            "Cited recall leaves out dataset '%s': answered by %s, which has no records to cite.",
            payload_.dataset_name,
            payload_.search_type,
        )

    shared_store_datasets = None
    items: list[SearchResultItem] = []
    handed_off_ids = {id(payload_) for payload_ in handed_off}
    for payload_ in prompted:
        if id(payload_) in handed_off_ids:
            continue
        evidence_dataset_ids = [payload_.dataset_id]
        if payload_.dataset_id is None:
            # Access control off: one search over the shared store, whose entities may
            # have evidence in any dataset the caller reads.
            if shared_store_datasets is None:
                shared_store_datasets = [
                    dataset.id
                    for dataset in await get_authorized_existing_datasets(None, "read", user)
                ]
            evidence_dataset_ids = shared_store_datasets
        items.extend(await build_cited_item(payload_, evidence_dataset_ids))
    return items


async def build_cited_item(
    payload_: SearchResultPayload, evidence_dataset_ids: list[UUID | None]
) -> list[SearchResultItem]:
    """One dataset's ``only_context`` hybrid result as its cited prompt, or [] when
    retrieval found nothing.

    The prompt is assembled as ``format_hybrid_context`` and ``build_completion_prompts``
    assemble regular mode's (the frame comes from ``payload_.context_frame``), with each
    record's tag line before its body. Entity sources are read from the edge evidence of
    ``evidence_dataset_ids``. Raises ``CitedTemplateNotSplittableError`` when the
    retriever's user template cannot be split around the context.
    """
    retrieved = payload_.result_object
    if not payload_.user_prompt:
        return []
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

    dataset_id = str(payload_.dataset_id) if payload_.dataset_id is not None else None
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
    entity_sources = await _entity_sources(entities, evidence_dataset_ids)

    records: list[CitedRecord] = []

    def cite(record_kind: str, record_id: str, body: str, full_body: str, **fields) -> str:
        """``body`` behind its tag line, recorded in ``records``. ``full_body`` is the body
        expand returns, which the version describes."""
        version = record_version(full_body)
        tag = record_tag(record_kind, dataset_id, record_id, version)
        records.append(
            CitedRecord(
                tag=tag,
                record_kind=record_kind,
                dataset_id=dataset_id,
                record_id=record_id,
                version=version,
                chars=len(body),
                full_chars=len(full_body),
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
                render_full_entity(entity),
                rank=rank,
                score=entity.get("score"),
                conflicts=list(entity.get("conflicts", [])),
                sources=entity_sources.get(entity["id"], []),
            )
        )

    fact_lines = []
    for rank, fact in enumerate(facts, start=1):
        line = f"- {fact['text']}"
        fact_lines.append(cite("fact", fact["id"], line, line, rank=rank, score=fact.get("score")))

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
    user: Any,
    retriever_kwargs: dict,
    llm_config: Any = None,
    embedding_config: Any = None,
) -> list[SearchResultItem]:
    """Full bodies for cited records, one result per requested item, in request order.

    Records are read by id alone, with no query. Items are grouped by dataset: read access
    is checked once per dataset and its stores are opened once; a repeated item repeats its
    row. A record the dataset no longer holds is ``missing``; a dataset the caller cannot
    read makes its records ``forbidden``. An item without a dataset (a tag of the shared
    store) is read from the shared store when access control is off and the caller reads
    some dataset, and is ``forbidden`` otherwise. ``retriever_kwargs`` are the settings
    that shape a passage's body (``include_references``, ``retriever_specific_config``).
    """
    unique_items = list(
        dict.fromkeys((item.dataset_id, item.record_kind, str(item.record_id)) for item in items)
    )
    versions = {
        (item.dataset_id, item.record_kind, str(item.record_id)): item.version for item in items
    }
    by_dataset: dict[UUID | None, list[tuple[str, str]]] = defaultdict(list)
    for dataset_id, record_kind, record_id in unique_items:
        by_dataset[dataset_id].append((record_kind, record_id))

    results: dict[tuple[UUID | None, str, str], SearchResultItem] = {}
    for dataset_id, records in by_dataset.items():
        dataset, evidence_dataset_ids, context = await _expand_scope(
            dataset_id, user, llm_config, embedding_config
        )
        if context is None:
            for record_kind, record_id in records:
                key = (dataset_id, record_kind, record_id)
                results[key] = _status_row(
                    dataset_id, None, record_kind, record_id, versions[key], "forbidden"
                )
            continue
        ids_by_kind: dict[str, list[str]] = defaultdict(list)
        for record_kind, record_id in records:
            ids_by_kind[record_kind].append(record_id)
        async with context:
            retriever = await get_search_type_retriever_instance(
                SearchType.HYBRID_COMPLETION, "", dataset=dataset, **retriever_kwargs
            )
            found = await retriever.expand_records(dict(ids_by_kind))
        results.update(
            await _expanded_rows(
                dataset_id,
                dataset.name if dataset is not None else None,
                records,
                found,
                versions,
                evidence_dataset_ids,
            )
        )
    keys = [(item.dataset_id, item.record_kind, str(item.record_id)) for item in items]
    return [results[key] for key in keys]


async def _expand_scope(dataset_id: UUID | None, user: Any, llm_config: Any, embedding_config: Any):
    """``(dataset, evidence dataset ids, database context)`` to expand records of
    ``dataset_id`` in, or a None context when the caller may not read them.

    ``dataset_id`` None names the shared store, which exists only with access control off;
    reading it needs read access to some dataset.
    """
    if dataset_id is not None:
        dataset = await _readable_dataset(dataset_id, user)
        if dataset is None:
            return None, [], None
        return (
            dataset,
            [dataset.id],
            set_database_global_context_variables(
                dataset.id,
                dataset.owner_id,
                llm_config=llm_config,
                embedding_config=embedding_config,
            ),
        )
    if backend_access_control_enabled():
        return None, [], None
    readable = await get_authorized_existing_datasets(None, "read", user)
    if not readable:
        return None, [], None
    return (
        None,
        [dataset.id for dataset in readable],
        set_database_global_context_variables(
            None, user.id, llm_config=llm_config, embedding_config=embedding_config
        ),
    )


async def _readable_dataset(dataset_id: UUID, user: Any):
    """The dataset when ``user`` may read it, else None (unknown and unreadable look alike)."""
    try:
        datasets = await get_authorized_existing_datasets([dataset_id], "read", user)
    except (PermissionDeniedError, DatasetNotFoundError):
        return None
    return datasets[0] if datasets else None


async def _expanded_rows(
    dataset_id: UUID | None,
    dataset_name: str | None,
    records: list[tuple[str, str]],
    found: dict,
    versions: dict,
    evidence_dataset_ids: list[UUID],
) -> dict:
    chunks = [found[key]["item"] for key in found if key[0] == "passage"]
    entities = [found[key]["item"] for key in found if key[0] == "entity"]
    created_at = await _created_at_by_data_id(
        [display_value(payload(chunk).get("document_id")) for chunk in chunks]
    )
    entity_sources = await _entity_sources(entities, evidence_dataset_ids)
    dataset = str(dataset_id) if dataset_id is not None else None

    rows = {}
    for record_kind, record_id in records:
        key = (dataset_id, record_kind, record_id)
        record = found.get((record_kind, record_id))
        if record is None or not record["body"]:
            rows[key] = _status_row(
                dataset_id, dataset_name, record_kind, record_id, versions[key], "missing"
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
            dataset_id=dataset,
            record_id=record_id,
            version=version,
            chars=len(body),
            full_chars=len(body),
            conflicts=record["conflicts"],
            sources=sources,
            status="available",
        )
        rows[key] = SearchResultItem(
            kind=search_result_kind(SearchType.HYBRID_COMPLETION),
            search_type=SearchType.HYBRID_COMPLETION,
            text=body,
            dataset_id=dataset,
            dataset_name=dataset_name,
            metadata=metadata.model_dump(mode="json"),
        )
    return rows


def _status_row(
    dataset_id: UUID | None,
    dataset_name: str | None,
    record_kind: str,
    record_id: str,
    version: str | None,
    status: str,
) -> SearchResultItem:
    """A ``missing`` or ``forbidden`` row. Its ``tag`` is the requested tag, so it can be
    sent again, or None when the item came without a version."""
    dataset = str(dataset_id) if dataset_id is not None else None
    metadata = CitedRecord(
        tag=record_tag(record_kind, dataset_id, record_id, version) if version else None,
        record_kind=record_kind,
        dataset_id=dataset,
        record_id=record_id,
        version=version,
        chars=0,
        status=status,
    )
    return SearchResultItem(
        kind=search_result_kind(SearchType.HYBRID_COMPLETION),
        search_type=SearchType.HYBRID_COMPLETION,
        text="",
        dataset_id=dataset,
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


async def _entity_sources(
    entities: list[dict], dataset_ids: list[UUID | None]
) -> dict[str, list[RecordSource]]:
    """The documents behind each entity's edges (all of them, the body its version
    describes), from the edge-evidence table of ``dataset_ids``.

    Only edges cognify extracted from document chunks have evidence; an entity known
    only through improve(), session bridging or the code graph has none.
    """
    dataset_ids = [UUID(str(dataset_id)) for dataset_id in dataset_ids if dataset_id]
    if not entities or not dataset_ids:
        return {}
    # An edge between two shown entities belongs to both, so its evidence is a source
    # of both.
    edge_entities: dict[UUID, list[str]] = defaultdict(list)
    for entity in entities:
        for edge in entity.get("all_edges", []):
            edge_object_id = display_value(edge.get("edge_object_id"))
            if edge_object_id and entity["id"] not in edge_entities[UUID(edge_object_id)]:
                edge_entities[UUID(edge_object_id)].append(entity["id"])
    if not edge_entities:
        return {}
    records = [
        record
        for dataset_id in dataset_ids
        for record in await get_edge_evidence_records(list(edge_entities), dataset_id)
    ]
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
