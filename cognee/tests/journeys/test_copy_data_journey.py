"""Journey: copy one document from a dataset into another, then forget it there.

Two datasets, one document each, sharing an entity (Wrenfield Quay). The first
document is copied into the second dataset: no LLM call, its chunks become searchable
there, the shared entity merges instead of duplicating, the second copy is a no-op,
``forget()`` in the destination removes exactly the copied document (by its source id,
which the destination row records as ``legacy_id``) and the source is untouched
throughout. A user without permission on the destination is refused.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.source_refs import parse_source_ref_key
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.data.exceptions import UnauthorizedDataAccessError
from cognee.modules.data.methods import get_data
from cognee.modules.provenance.edge_evidence.models import ProvenanceEdgeEvidence
from cognee.modules.search.types import SearchType
from cognee.tests.journeys import _support

SOURCE, DESTINATION = "copy_source", "copy_destination"
DOC_A = (
    "Title: Harbour Lantern Society\n\nThe Harbour Lantern Society is led by Priya Vasquez "
    "and meets at Wrenfield Quay every Tuesday."
)
DOC_B = (
    "Title: Wrenfield Rowing Club\n\nThe Wrenfield Rowing Club trains at Wrenfield Quay "
    "under coach Tobias Lindqvist."
)


async def _node_ids(dataset_id, user) -> set[str]:
    async with set_database_global_context_variables(dataset_id, user.id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
    return {str(node_id) for node_id, _ in nodes}


async def _owners_of_nodes(dataset_id, user) -> dict[str, set]:
    """node id -> the data ids its source refs name, in one dataset."""
    async with set_database_global_context_variables(dataset_id, user.id):
        refs = await (await get_graph_engine()).find_node_source_refs_by_dataset(str(dataset_id))
    return {
        str(node_id): {parse_source_ref_key(key).data_id for key in keys}
        for node_id, keys in refs.items()
    }


async def _chunk_text(dataset: str, query: str) -> str:
    results = await cognee.search(
        query_text=query, query_type=SearchType.CHUNKS, datasets=[dataset]
    )
    return _support.content_text(results)


async def _evidence_links(dataset_id, data_id) -> set[tuple]:
    """The distinct chunk -> edge support links recorded for one document. The source
    may hold one link several times, once per pipeline run that saw it; the copy
    carries no run, so it holds each link once."""
    async with get_relational_engine().get_async_session() as session:
        rows = await session.execute(
            select(
                ProvenanceEdgeEvidence.chunk_id,
                ProvenanceEdgeEvidence.edge_id,
                ProvenanceEdgeEvidence.evidence_kind,
            ).where(
                ProvenanceEdgeEvidence.dataset_id == dataset_id,
                ProvenanceEdgeEvidence.data_id == data_id,
            )
        )
    return set(rows.all())


@pytest.mark.journey
@pytest.mark.asyncio
async def test_copy_data_moves_a_document_between_datasets(clean_env, default_user, mock_llm):
    await cognee.remember(DOC_A, dataset_name=SOURCE, self_improvement=False)
    await cognee.remember(DOC_B, dataset_name=DESTINATION, self_improvement=False)
    source_id = await _support.dataset_id_by_name(SOURCE, default_user)
    destination_id = await _support.dataset_id_by_name(DESTINATION, default_user)
    (document,) = await cognee.datasets.list_data(source_id, default_user)

    source_before = await _support.snapshot_dataset(source_id, default_user)
    source_nodes = await _node_ids(source_id, default_user)
    destination_nodes_before = await _node_ids(destination_id, default_user)
    assert source_nodes & destination_nodes_before, "the two documents must share an entity"
    llm_calls_before = len(mock_llm.calls)

    # --- copy -------------------------------------------------------------------
    receipt = await cognee.datasets.copy_data(
        dataset_id=source_id,
        data_id=document.id,
        target_dataset_id=destination_id,
        user=default_user,
    )

    assert receipt["status"] == "success"
    assert receipt["data_record_created"] is True
    assert receipt["copied_nodes"] == len(source_nodes)
    assert receipt["copied_edges"] > 0
    assert receipt["copied_vectors"] > 0
    source_links = await _evidence_links(source_id, document.id)
    assert source_links
    assert receipt["copied_evidence"] == len(source_links)
    assert len(mock_llm.calls) == llm_calls_before, "a copy never calls the LLM"

    # the destination row: new id, source id kept as legacy_id, cognified for the destination
    copied_row = await get_data(default_user.id, receipt["data_id"], destination_id)
    assert copied_row.legacy_id == document.id
    assert copied_row.raw_data_location == document.raw_data_location
    assert len(await cognee.datasets.list_data(destination_id, default_user)) == 2

    # graph: union of both documents, shared entity merged, nothing duplicated
    assert await _node_ids(destination_id, default_user) == source_nodes | destination_nodes_before
    # provenance: a shared node now names both documents, a copied-only node names the
    # copy, and a destination-only node still names only its own document
    (doc_b,) = [
        row
        for row in await cognee.datasets.list_data(destination_id, default_user)
        if row.id != receipt["data_id"]
    ]
    owners = await _owners_of_nodes(destination_id, default_user)
    shared = source_nodes & destination_nodes_before
    for node_id in shared:
        assert owners[node_id] == {doc_b.id, receipt["data_id"]}
    for node_id in source_nodes - destination_nodes_before:
        assert owners[node_id] == {receipt["data_id"]}
    for node_id in destination_nodes_before - source_nodes:
        assert owners[node_id] == {doc_b.id}
    assert "lantern" in (await _chunk_text(DESTINATION, "Harbour Lantern Society")).lower()
    assert await _evidence_links(destination_id, receipt["data_id"]) == source_links
    assert await _support.snapshot_dataset(source_id, default_user) == source_before

    # --- copying again reuses the row and changes nothing ----------------------------
    destination_after_copy = await _support.snapshot_dataset(destination_id, default_user)
    again = await cognee.datasets.copy_data(
        dataset_id=source_id,
        data_id=document.id,
        target_dataset_id=destination_id,
        user=default_user,
    )
    assert again["data_id"] == receipt["data_id"]
    assert again["data_record_created"] is False
    assert await _support.snapshot_dataset(destination_id, default_user) == destination_after_copy

    # --- forget the copy in the destination by its SOURCE id (legacy_id resolves) ----
    await cognee.forget(data_id=document.id, dataset_id=destination_id, user=default_user)

    assert await _node_ids(destination_id, default_user) == destination_nodes_before
    owners = await _owners_of_nodes(destination_id, default_user)
    assert all(owners[node_id] == {doc_b.id} for node_id in destination_nodes_before)
    assert "lantern" not in (await _chunk_text(DESTINATION, "Harbour Lantern Society")).lower()
    assert len(await cognee.datasets.list_data(destination_id, default_user)) == 1
    assert not await _evidence_links(destination_id, receipt["data_id"])
    assert "lantern" in (await _chunk_text(SOURCE, "Harbour Lantern Society")).lower()
    assert await _support.snapshot_dataset(source_id, default_user) == source_before


@pytest.mark.journey
@pytest.mark.asyncio
async def test_copy_data_refuses_without_permission_or_into_itself(clean_env, default_user):
    from cognee.modules.users.methods import create_user

    await cognee.remember(DOC_A, dataset_name=SOURCE, self_improvement=False)
    await cognee.remember(DOC_B, dataset_name=DESTINATION, self_improvement=False)
    source_id = await _support.dataset_id_by_name(SOURCE, default_user)
    destination_id = await _support.dataset_id_by_name(DESTINATION, default_user)
    (document,) = await cognee.datasets.list_data(source_id, default_user)

    with pytest.raises(CogneeValidationError, match="both source and destination"):
        await cognee.datasets.copy_data(
            dataset_id=source_id,
            data_id=document.id,
            target_dataset_id=source_id,
            user=default_user,
        )

    stranger = await create_user(f"stranger-{uuid4().hex}@example.com", "secret")
    with pytest.raises(UnauthorizedDataAccessError):
        await cognee.datasets.copy_data(
            dataset_id=source_id,
            data_id=document.id,
            target_dataset_id=destination_id,
            user=stranger,
        )

    with pytest.raises(UnauthorizedDataAccessError, match="not accessible in dataset"):
        await cognee.datasets.copy_data(
            dataset_id=source_id,
            data_id=uuid4(),
            target_dataset_id=destination_id,
            user=default_user,
        )
    assert len(await cognee.datasets.list_data(destination_id, default_user)) == 1


async def _remember_both(default_user):
    await cognee.remember(DOC_A, dataset_name=SOURCE, self_improvement=False)
    await cognee.remember(DOC_B, dataset_name=DESTINATION, self_improvement=False)
    source_id = await _support.dataset_id_by_name(SOURCE, default_user)
    destination_id = await _support.dataset_id_by_name(DESTINATION, default_user)
    (document,) = await cognee.datasets.list_data(source_id, default_user)
    return source_id, destination_id, document


@pytest.mark.journey
@pytest.mark.asyncio
async def test_copy_data_leaves_unconfirmed_evidence_behind(clean_env, default_user):
    """An evidence row whose pipeline run never completed is ignored by the source's
    own lookup; the copy must not turn it into a confirmed citation."""
    source_id, destination_id, document = await _remember_both(default_user)
    confirmed = await _evidence_links(source_id, document.id)
    assert confirmed
    phantom_chunk, phantom_edge = uuid4(), uuid4()
    async with get_relational_engine().get_async_session() as session:
        session.add(
            ProvenanceEdgeEvidence(
                user_id=default_user.id,
                dataset_id=source_id,
                data_id=document.id,
                pipeline_run_id=uuid4(),  # no such run: never completed
                chunk_id=phantom_chunk,
                chunk_index=0,
                edge_id=phantom_edge,
                source_node_id=uuid4(),
                destination_node_id=uuid4(),
                relationship_name="phantom",
                created_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    receipt = await cognee.datasets.copy_data(
        dataset_id=source_id,
        data_id=document.id,
        target_dataset_id=destination_id,
        user=default_user,
    )

    assert receipt["copied_evidence"] == len(confirmed)
    assert await _evidence_links(destination_id, receipt["data_id"]) == confirmed


@pytest.mark.journey
@pytest.mark.asyncio
async def test_copy_data_takes_a_failed_first_copy_back_out(clean_env, default_user, monkeypatch):
    """A write that fails after the graph went in must not leave a row stamped
    cognified in front of a partial copy: the destination ends as it began."""
    source_id, destination_id, document = await _remember_both(default_user)
    before = await _support.snapshot_dataset(destination_id, default_user)
    nodes_before = await _node_ids(destination_id, default_user)

    async with set_database_global_context_variables(destination_id, default_user.id):
        adapter_class = (await get_vector_engine_async()).__class__

    async def refuse(self, *args, **kwargs):
        raise RuntimeError("vector store refused the write")

    monkeypatch.setattr(adapter_class, "upsert_raw_vectors", refuse)
    with pytest.raises(RuntimeError, match="refused the write"):
        await cognee.datasets.copy_data(
            dataset_id=source_id,
            data_id=document.id,
            target_dataset_id=destination_id,
            user=default_user,
        )
    monkeypatch.undo()

    assert await _support.snapshot_dataset(destination_id, default_user) == before
    assert await _node_ids(destination_id, default_user) == nodes_before
    assert len(await cognee.datasets.list_data(destination_id, default_user)) == 1
    assert "lantern" not in (await _chunk_text(DESTINATION, "Harbour Lantern Society")).lower()

    # and the same copy succeeds afterwards, as a first copy
    receipt = await cognee.datasets.copy_data(
        dataset_id=source_id,
        data_id=document.id,
        target_dataset_id=destination_id,
        user=default_user,
    )
    assert receipt["data_record_created"] is True
    assert "lantern" in (await _chunk_text(DESTINATION, "Harbour Lantern Society")).lower()


DOC_QUAY_1902 = (
    "Title: Wrenfield Quay History\n\nWrenfield Quay was built in 1902 by the Harbour "
    "Lantern Society, and it has exactly three mooring berths."
)
DOC_QUAY_1910 = (
    "Title: Wrenfield Quay Records\n\nWrenfield Quay was built in 1910 by the Wrenfield "
    "Rowing Club, and it has exactly five mooring berths."
)


async def _node_types(dataset_id, user) -> set[str]:
    async with set_database_global_context_variables(dataset_id, user.id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
    return {str(props.get("type")) for _, props in nodes}


@pytest.mark.journey
@pytest.mark.asyncio
async def test_copy_data_can_review_the_destination_for_conflicts(clean_env, default_user):
    """``review_conflicts=True`` runs cognee's conflict review over the destination
    after the copy. With a real LLM (``COGNEE_JOURNEY_MODE=llm``) the two documents'
    contradicting facts about Wrenfield Quay become a ``FactConflict`` node in the
    destination; under the mock LLM the stage either runs on canned answers or is
    skipped for want of a key, and the receipt carries its result either way."""
    await cognee.remember(DOC_QUAY_1902, dataset_name=SOURCE, self_improvement=False)
    await cognee.remember(DOC_QUAY_1910, dataset_name=DESTINATION, self_improvement=False)
    source_id = await _support.dataset_id_by_name(SOURCE, default_user)
    destination_id = await _support.dataset_id_by_name(DESTINATION, default_user)
    (document,) = await cognee.datasets.list_data(source_id, default_user)
    assert "FactConflict" not in await _node_types(destination_id, default_user)

    receipt = await cognee.datasets.copy_data(
        dataset_id=source_id,
        data_id=document.id,
        target_dataset_id=destination_id,
        user=default_user,
        review_conflicts=True,
    )

    review = receipt["conflict_review"]
    assert review["stage"] == "review_conflicts"
    assert "1902" in (await _chunk_text(DESTINATION, "When was Wrenfield Quay built")).lower()
    if _support.IS_MOCK:
        # Whether the stage runs on the mock depends on the environment: with no
        # usable key it is skipped, with one the mock answers the review's prompts.
        # Either way the receipt carries the stage, and nothing errored.
        assert review["status"] in ("completed", "already_completed", "skipped"), review
        if review["status"] == "skipped":
            assert review["reason"] == "no_llm_configured"
    else:
        assert review["status"] == "completed", review
        assert "FactConflict" in await _node_types(destination_id, default_user)
