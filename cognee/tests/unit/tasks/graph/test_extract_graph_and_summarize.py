import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.infrastructure.engine import DataPoint, Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import NodeSet
from cognee.modules.graph.utils import construct_data_points_and_edges
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.summarization.models import TextSummary

task_module = importlib.import_module("cognee.tasks.graph.extract_graph_and_summarize")


def _document():
    return TextDocument(
        name="notes.txt",
        raw_data_location="/tmp/notes.txt",
        external_metadata="",
        mime_type="text/plain",
    )


def _chunk(text, belongs_to_set=None, chunk_index=0):
    return DocumentChunk(
        text=text,
        chunk_size=len(text.split()),
        chunk_index=chunk_index,
        cut_type="sentence_end",
        is_part_of=_document(),
        contains=[],
        belongs_to_set=belongs_to_set,
    )


def _patch(monkeypatch, extract):
    monkeypatch.setattr(task_module, "extract_graph_from_data", extract)
    summarize = AsyncMock(return_value=["llm-summary"])
    monkeypatch.setattr(task_module, "summarize_text", summarize)
    return summarize


@pytest.mark.asyncio
async def test_llm_returns_the_summarize_text_list(monkeypatch):
    chunks = [_chunk("Acme agreed to buy Beta Corp.")]
    summarize = _patch(monkeypatch, AsyncMock(return_value=chunks))

    result = await task_module.extract_graph_and_summarize(
        chunks, KnowledgeGraph, summary_method="llm"
    )

    summarize.assert_awaited_once()
    assert result == ["llm-summary"]


@pytest.mark.asyncio
async def test_omitted_summary_method_reads_the_config(monkeypatch):
    chunks = [_chunk("Acme agreed to buy Beta Corp.")]
    summarize = _patch(monkeypatch, AsyncMock(return_value=chunks))
    monkeypatch.setattr(
        task_module,
        "get_cognify_config",
        lambda: SimpleNamespace(summary_method="from_extraction"),
    )

    result = await task_module.extract_graph_and_summarize(chunks, KnowledgeGraph)

    summarize.assert_not_awaited()
    assert result == chunks


def _deal_graph():
    nodes = [
        Node(id="acme", name="Acme", type="Company", description="Buyer"),
        Node(id="beta", name="Beta Corp", type="Company", description="Target"),
        Node(id="carol", name="Carol Diaz", type="Person", description="Founder"),
    ]
    buy = KGEdge(
        source_node_id="acme",
        target_node_id="beta",
        relationship_name="agreed_to_buy",
        description="Acme agreed to buy Beta Corp.",
    )
    edges = [
        buy,
        KGEdge(
            source_node_id="carol",
            target_node_id="beta",
            relationship_name="founded",
            description="Carol Diaz founded Beta Corp.",
        ),
        buy,
        KGEdge(
            source_node_id="carol",
            target_node_id="acme",
            relationship_name="works_with",
        ),
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)


@pytest.mark.asyncio
async def test_from_extraction_summarizes_knowledge_graph_relations(monkeypatch):
    chunk_1 = _chunk("Acme agreed to buy Beta Corp.", belongs_to_set=["KEEP"], chunk_index=0)
    chunk_2 = _chunk("An unrelated sentence.", chunk_index=1)
    graphs = [
        _deal_graph(),
        KnowledgeGraph(
            nodes=[Node(id="solo", name="Solo", type="Person", description="Alone")],
            edges=[],
        ),
    ]

    async def extract(data_chunks, graph_model, **kwargs):
        construct_data_points_and_edges(data_chunks, graphs)
        return data_chunks

    summarize = _patch(monkeypatch, extract)

    result = await task_module.extract_graph_and_summarize(
        [chunk_1, chunk_2], KnowledgeGraph, summary_method="from_extraction"
    )

    summarize.assert_not_awaited()
    summary, bare_chunk = result
    assert bare_chunk is chunk_2
    assert isinstance(summary, TextSummary)
    assert summary.text == (
        "company: acme, beta corp\n"
        "person: carol diaz\n"
        "Acme agreed to buy Beta Corp.\n"
        "Carol Diaz founded Beta Corp.\n"
        "carol diaz works with acme."
    )
    assert summary.id == uuid5(chunk_1.id, "TextSummary")
    assert summary.source_chunk_id == str(chunk_1.id)
    assert summary.made_from is chunk_1
    assert summary.belongs_to_set == ["KEEP"]
    blank = [edge for edge in chunk_1._provenance_edges if edge[3]["edge_text"] is None]
    assert len(chunk_1._provenance_edges) == 4
    assert len(blank) == 1
    assert blank[0][2] == "works_with"


class Company(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class Person(DataPoint):
    name: str
    works_for: Company
    reports_to: list[Edge["Person", "Person"]] = []
    metadata: dict = {"index_fields": ["name"]}


def _maya():
    acme = Company(name="Acme")
    priya = Person(name="Priya", works_for=acme)
    maya = Person(name="Maya", works_for=acme)
    # Set after init. Passing the Edge into the constructor builds the field's
    # unresolved Edge["Person", "Person"], which cannot be stored. The pipeline
    # assigns a plain Edge the same way.
    maya.reports_to = [Edge(target=priya, edge_text="Maya reports to Priya.")]
    return maya


@pytest.mark.asyncio
async def test_from_extraction_summarizes_a_custom_graph_model(monkeypatch):
    chunk = _chunk(
        "Maya works at Acme and reports to Priya.",
        belongs_to_set=[NodeSet(name="projectX")],
    )

    async def extract(data_chunks, graph_model, **kwargs):
        for item in data_chunks:
            item.contains = _maya()
        return data_chunks

    summarize = _patch(monkeypatch, extract)

    result = await task_module.extract_graph_and_summarize(
        [chunk], Person, summary_method="from_extraction"
    )

    summarize.assert_not_awaited()
    assert result[0].text == (
        "Person: Maya, Priya\n"
        "Company: Acme\n"
        "Maya works for Acme.\n"
        "Maya reports to Priya.\n"
        "Priya works for Acme."
    )


def _dataset_ctx(dataset_id):
    return SimpleNamespace(dataset=SimpleNamespace(id=dataset_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("passed_prompt", [None, ""])
async def test_missing_prompt_uses_the_dataset_prompt(monkeypatch, passed_prompt):
    chunks = [_chunk("Acme agreed to buy Beta Corp.")]
    extract = AsyncMock(return_value=chunks)
    _patch(monkeypatch, extract)
    stored = AsyncMock(return_value="Extract companies and deals.")
    monkeypatch.setattr(task_module, "get_dataset_custom_prompt", stored)
    dataset_id = uuid5(NAMESPACE_OID, "ds")

    await task_module.extract_graph_and_summarize(
        chunks,
        KnowledgeGraph,
        custom_prompt=passed_prompt,
        ctx=_dataset_ctx(dataset_id),
        summary_method="llm",
    )

    stored.assert_awaited_once_with(dataset_id)
    assert extract.await_args.kwargs["custom_prompt"] == "Extract companies and deals."


@pytest.mark.asyncio
async def test_passed_prompt_wins_over_the_dataset_prompt(monkeypatch):
    chunks = [_chunk("Acme agreed to buy Beta Corp.")]
    extract = AsyncMock(return_value=chunks)
    _patch(monkeypatch, extract)
    stored = AsyncMock(return_value="Extract companies and deals.")
    monkeypatch.setattr(task_module, "get_dataset_custom_prompt", stored)

    await task_module.extract_graph_and_summarize(
        chunks,
        KnowledgeGraph,
        custom_prompt="Extract people only.",
        ctx=_dataset_ctx(uuid5(NAMESPACE_OID, "ds")),
        summary_method="llm",
    )

    stored.assert_not_awaited()
    assert extract.await_args.kwargs["custom_prompt"] == "Extract people only."


@pytest.mark.asyncio
async def test_no_dataset_in_context_keeps_the_built_in_prompt(monkeypatch):
    chunks = [_chunk("Acme agreed to buy Beta Corp.")]
    extract = AsyncMock(return_value=chunks)
    _patch(monkeypatch, extract)
    stored = AsyncMock(return_value="Extract companies and deals.")
    monkeypatch.setattr(task_module, "get_dataset_custom_prompt", stored)

    await task_module.extract_graph_and_summarize(chunks, KnowledgeGraph, summary_method="llm")

    stored.assert_not_awaited()
    assert extract.await_args.kwargs["custom_prompt"] is None
