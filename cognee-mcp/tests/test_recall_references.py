"""Recall can name the data item behind each hit, so forget can delete just that item."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp import Client
from src import server
from src.cognee_client import CogneeClient
from src.server_utils import (
    MAX_RECALL_REFS,
    format_recall_body,
    format_recall_results,
    recall_item_dataset,
    recall_item_refs,
)

DATA_A = "11111111-1111-1111-1111-111111111111"
DATA_B = "22222222-2222-2222-2222-222222222222"
DATASET = "33333333-3333-3333-3333-333333333333"
OTHER_DATASET = "44444444-4444-4444-4444-444444444444"


def _chunk_hit(text: str, data_id: str, document_name: str = "notes.txt") -> dict:
    return {
        "source": "graph",
        "kind": "chunk",
        "text": text,
        "dataset_id": DATASET,
        "dataset_name": "main_dataset",
        "metadata": {
            "data_id": data_id,
            "chunk_id": "c-" + data_id[:4],
            "document_name": document_name,
        },
    }


def _segment(data_id: str, dataset_id: str | None = DATASET, document_name: str | None = None):
    return {
        "kind": "segment",
        "artifact_id": "chunk-" + data_id[:4],
        "data_id": data_id,
        "dataset_id": dataset_id,
        "document_name": document_name,
    }


def test_chunk_hit_refs_come_from_its_own_metadata():
    assert recall_item_refs(_chunk_hit("A", DATA_A)) == [
        {"data_id": DATA_A, "dataset_id": DATASET, "document": "notes.txt"}
    ]


def test_completion_refs_come_from_segment_evidence_deduplicated():
    hit = {
        "source": "graph",
        "text": "The answer.",
        "dataset_id": DATASET,
        "metadata": {
            "evidence": [
                _segment(DATA_A, document_name="a.txt"),
                _segment(DATA_A, document_name="a.txt"),  # second chunk of the same item
                {"kind": "graph_node", "artifact_id": "n1", "dataset_id": DATASET},
                _segment(DATA_B, dataset_id=None),  # falls back to the hit's dataset
                _segment(DATA_B, dataset_id=OTHER_DATASET),  # same item id, other dataset
            ]
        },
    }
    assert recall_item_refs(hit) == [
        {"data_id": DATA_A, "dataset_id": DATASET, "document": "a.txt"},
        {"data_id": DATA_B, "dataset_id": DATASET},
        {"data_id": DATA_B, "dataset_id": OTHER_DATASET},
    ]


def test_refs_are_capped():
    evidence = [_segment(f"{i:08d}-0000-0000-0000-000000000000") for i in range(20)]
    hit = {"text": "x", "dataset_id": DATASET, "metadata": {"evidence": evidence}}
    assert len(recall_item_refs(hit)) == MAX_RECALL_REFS


@pytest.mark.parametrize(
    "hit",
    [
        {"source": "session", "answer": "cached answer"},
        {"source": "graph", "text": "no dataset", "metadata": {"data_id": DATA_A}},
        {
            "source": "graph",
            "text": "node only",
            "metadata": {"evidence": [{"kind": "graph_node"}]},
        },
        "a bare string",
    ],
)
def test_hits_without_a_forgettable_item_have_no_refs(hit):
    assert recall_item_refs(hit) == []


def test_pydantic_recall_entries_are_read_like_dicts():
    from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry

    entry = ResponseGraphEntry(
        source="graph",
        kind="chunk",
        search_type="CHUNKS",
        text="A",
        dataset_id=DATASET,
        metadata={"data_id": DATA_A},
    )
    assert recall_item_refs(entry) == [{"data_id": DATA_A, "dataset_id": DATASET}]


def test_body_without_references_is_unchanged():
    payload = [_chunk_hit("A", DATA_A), {"source": "session", "answer": "B"}]
    assert format_recall_body(payload) == "[graph] A\n\n[session] B"
    assert format_recall_results(payload) == format_recall_results(
        payload, include_references=False
    )


def test_body_with_references_puts_refs_under_each_hit():
    payload = [
        _chunk_hit("recovery phrase", DATA_A),
        {"source": "session", "answer": "B"},
        _chunk_hit("lunch on Fridays", DATA_B, document_name="lunch\nnotes.txt"),
    ]
    assert format_recall_body(payload, include_references=True) == (
        "[graph] recovery phrase\n"
        f"  refs: data_id={DATA_A}, dataset_id={DATASET}, document=notes.txt\n\n"
        "[session] B\n\n"
        "[graph] lunch on Fridays\n"
        f"  refs: data_id={DATA_B}, dataset_id={DATASET}, document=lunch notes.txt"
    )
    rendered = format_recall_results(payload, include_references=True)
    assert rendered.splitlines()[0] == "3 memories found (2 from main_dataset, 1 from sessions)"


def test_evidence_is_taken_best_rank_first():
    ids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(4)]
    evidence = [
        {**_segment(ids[0]), "rank": 3},
        {**_segment(ids[1]), "rank": None},
        {**_segment(ids[2]), "rank": 0},
        {**_segment(ids[3]), "rank": 1},
    ]
    hit = {"text": "x", "dataset_id": DATASET, "metadata": {"evidence": evidence}}
    assert [ref["data_id"] for ref in recall_item_refs(hit)] == [ids[2], ids[3], ids[0], ids[1]]


def _hybrid_hit(dataset_id: str | None = DATASET) -> dict:
    # The default hybrid answer: no structured evidence, data_ids cited in the text.
    return {
        "source": "graph",
        "kind": "graph_completion",
        "text": f'amber-falcon-42\n\nEvidence:\n- chunk 1 of document t (data_id: {DATA_A}): "..."',
        "dataset_id": dataset_id,
        "dataset_name": "main_dataset",
        "metadata": {},
    }


def test_hit_without_refs_gets_a_dataset_line():
    hit = _hybrid_hit()
    assert recall_item_dataset(hit) == f"dataset: main_dataset (dataset_id={DATASET})"
    body = format_recall_body([hit], include_references=True)
    assert body == f"[graph] {hit['text']}\n  dataset: main_dataset (dataset_id={DATASET})"
    assert format_recall_body([hit]) == f"[graph] {hit['text']}"


def test_no_dataset_line_without_a_dataset_id():
    # Access control off and an unscoped recall: search knows no dataset.
    hit = _hybrid_hit(dataset_id=None)
    assert recall_item_dataset(hit) is None
    assert format_recall_body([hit], include_references=True) == f"[graph] {hit['text']}"


@pytest.mark.asyncio
async def test_recall_tool_forwards_flag_and_renders_refs(monkeypatch):
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=[_chunk_hit("A", DATA_A)]),
        get_recall_state=AsyncMock(),
    )
    monkeypatch.setattr(server, "cognee_client", fake)

    result = await server.recall("q", search_type="CHUNKS", include_references=True)
    assert fake.recall.call_args.kwargs["include_references"] is True
    assert f"refs: data_id={DATA_A}, dataset_id={DATASET}" in result[0].text

    result = await server.recall("q", search_type="CHUNKS")
    assert fake.recall.call_args.kwargs["include_references"] is False
    assert "refs:" not in result[0].text


@pytest.mark.asyncio
async def test_recall_tool_schema_exposes_include_references():
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    props = tools["recall"].inputSchema.get("properties", {})
    assert props["include_references"]["default"] is False


@pytest.mark.asyncio
async def test_direct_client_passes_include_references_only_when_set():
    client = CogneeClient()
    client.cognee = SimpleNamespace(recall=AsyncMock(return_value=[]))

    await client.recall("q", datasets=["d"], system_prompt="s", include_references=True)
    assert client.cognee.recall.call_args.kwargs["include_references"] is True

    await client.recall("q", datasets=["d"], system_prompt="s")
    assert "include_references" not in client.cognee.recall.call_args.kwargs


@pytest.mark.asyncio
async def test_api_client_sends_include_references_only_when_set():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.recall("q", datasets=["d"], system_prompt="s", include_references=True)
        await client.recall("q", datasets=["d"], system_prompt="s")
    finally:
        await client.close()

    assert json.loads(requests[0].content.decode())["include_references"] is True
    assert "include_references" not in json.loads(requests[1].content.decode())
