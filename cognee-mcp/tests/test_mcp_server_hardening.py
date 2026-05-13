import asyncio
import hashlib
import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest


MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

CogneeClient = importlib.import_module("src.cognee_client").CogneeClient
server_utils = importlib.import_module("src.server_utils")
retrieval_utils = importlib.import_module("src.retrieval_utils")
format_recall_results = server_utils.format_recall_results
format_search_results = server_utils.format_search_results
normalize_delete_mode = server_utils.normalize_delete_mode
parse_cognify_data = server_utils.parse_cognify_data
validate_cognify_file_paths = server_utils.validate_cognify_file_paths
validate_top_k = server_utils.validate_top_k
get_chunk_neighbors_from_graph = retrieval_utils.get_chunk_neighbors_from_graph
get_document_from_graph = retrieval_utils.get_document_from_graph


def test_parse_cognify_data_accepts_plain_text():
    parsed = parse_cognify_data("plain text")

    assert parsed.items == ["plain text"]
    assert parsed.is_batch is False


def test_parse_cognify_data_accepts_json_batch():
    parsed = parse_cognify_data(json.dumps(["/tmp/a.txt", "inline memory"]))

    assert parsed.items == ["/tmp/a.txt", "inline memory"]
    assert parsed.is_batch is True


@pytest.mark.parametrize("payload", ["[]", "[1]", '[""]', "[1, 2]"])
def test_parse_cognify_data_rejects_invalid_batches(payload):
    with pytest.raises(ValueError):
        parse_cognify_data(payload)


def test_parse_cognify_data_preserves_plain_text_starting_with_bracket():
    parsed = parse_cognify_data("[note: inline memory")

    assert parsed.items == ["[note: inline memory"]
    assert parsed.is_batch is False


def test_validate_cognify_file_paths_reports_batch_index():
    error = validate_cognify_file_paths(
        ["/missing/file.txt"],
        path_exists=lambda _: False,
    )

    assert "File not found: /missing/file.txt" in error

    batch_error = validate_cognify_file_paths(
        ["inline text", "/missing/file.txt"],
        path_exists=lambda _: False,
    )

    assert "Invalid batch item at index 1" in batch_error
    assert "File not found: /missing/file.txt" in batch_error


def test_validate_top_k_and_delete_mode():
    assert validate_top_k("3") == 3
    assert normalize_delete_mode(" HARD ") == "hard"

    with pytest.raises(ValueError):
        validate_top_k(0)
    with pytest.raises(ValueError):
        validate_top_k(101)
    with pytest.raises(ValueError):
        normalize_delete_mode("unsafe")


def test_format_search_results_handles_envelope_and_completion_rows():
    rendered = format_search_results(
        {
            "query": "what matters?",
            "results": [
                {"dataset_name": "alpha", "search_result": ["first answer", "second answer"]},
                {"dataset_name": "beta", "text": "third answer"},
            ],
        },
        "GRAPH_COMPLETION",
    )

    assert "[alpha] first answer" in rendered
    assert "[alpha] second answer" in rendered
    assert "[beta] third answer" in rendered


def test_format_recall_results_handles_normalized_rows():
    rendered = format_recall_results(
        {
            "results": [
                {"source": "session", "text": "cached answer"},
                {"_source": "graph", "answer": "graph answer"},
            ]
        }
    )

    assert "[session] cached answer" in rendered
    assert "[graph] graph answer" in rendered


@pytest.mark.asyncio
async def test_mcp_exposes_only_memory_tools():
    import src.server as server

    tools = await server.mcp.list_tools()

    assert {tool.name for tool in tools} == {"remember", "recall", "forget"}


@pytest.mark.asyncio
async def test_cognee_client_api_add_uses_content_addressed_filename():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    content = "first memory"
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()

    try:
        await client.add(content, dataset_name="ds")
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/add"
    body = requests[0].content.decode()
    assert f'filename="text_{digest}.txt"' in body
    assert "data.txt" not in body


@pytest.mark.asyncio
async def test_cognee_client_api_remember_sends_session_id():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = CogneeClient(api_url="http://cognee.local", api_token="token")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    content = "hello"
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()

    try:
        await client.remember(content, dataset_name="ds", session_id="session-1")
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/remember"
    body = requests[0].content.decode()
    assert f'filename="text_{digest}.txt"' in body
    assert "data.txt" not in body
    assert 'name="session_id"' in body
    assert "session-1" in body


@pytest.mark.asyncio
async def test_cognee_client_api_recall_sends_session_id_and_null_search_type():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.recall("hello", session_id="session-1", top_k=5)
    finally:
        await client.close()

    payload = json.loads(requests[0].content.decode())
    assert requests[0].url.path == "/api/v1/recall"
    assert payload["session_id"] == "session-1"
    assert payload["search_type"] is None
    assert payload["top_k"] == 5


@pytest.mark.asyncio
async def test_cognee_client_api_delete_uses_mode_aware_endpoint():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "success"})

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.delete(
            data_id="00000000-0000-0000-0000-000000000001",
            dataset_id="00000000-0000-0000-0000-000000000002",
            mode="hard",
        )
    finally:
        await client.close()

    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/api/v1/delete"
    assert requests[0].url.params["data_id"] == "00000000-0000-0000-0000-000000000001"
    assert requests[0].url.params["dataset_id"] == "00000000-0000-0000-0000-000000000002"
    assert requests[0].url.params["mode"] == "hard"


@pytest.mark.asyncio
async def test_cognify_tool_batches_add_calls(monkeypatch, tmp_path):
    import src.server as server

    data_file = tmp_path / "memory.txt"
    data_file.write_text("memory", encoding="utf-8")

    class FakeClient:
        use_api = False

        def __init__(self):
            self.added = []
            self.cognified = None

        async def add(self, data, dataset_name="main_dataset"):
            self.added.append((data, dataset_name))

        async def cognify(self, datasets=None, custom_prompt=None, graph_model=None):
            self.cognified = {
                "datasets": datasets,
                "custom_prompt": custom_prompt,
                "graph_model": graph_model,
            }

    fake_client = FakeClient()
    created_tasks = []
    original_create_task = asyncio.create_task

    def capture_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(server, "cognee_client", fake_client)
    monkeypatch.setattr(server.asyncio, "create_task", capture_task)

    result = await server.cognify(
        json.dumps([str(data_file), "inline memory"]),
        dataset_name="batch_ds",
        custom_prompt="extract carefully",
    )

    assert "Queued 2 item(s)" in result[0].text
    await created_tasks[0]
    assert fake_client.added == [(str(data_file), "batch_ds"), ("inline memory", "batch_ds")]
    assert fake_client.cognified == {
        "datasets": ["batch_ds"],
        "custom_prompt": "extract carefully",
        "graph_model": None,
    }


class FakeGraph:
    def __init__(self, nodes=None, connections=None, subgraphs=None):
        self.nodes = nodes or {}
        self.connections = connections or {}
        self.subgraphs = subgraphs or {}

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def get_connections(self, node_id):
        return self.connections.get(node_id, [])

    async def get_document_subgraph(self, document_id):
        return self.subgraphs.get(document_id)


def _document_node():
    return {
        "id": "doc-1",
        "name": "Guide",
        "type": "TextDocument",
        "raw_data_location": "/tmp/guide.txt",
        "mime_type": "text/plain",
    }


def _chunk_node(chunk_id, index, text):
    return {
        "id": chunk_id,
        "type": "DocumentChunk",
        "chunk_index": index,
        "text": text,
        "chunk_size": len(text.split()),
    }


@pytest.mark.asyncio
async def test_get_document_from_graph_uses_subgraph_sorts_and_truncates():
    graph = FakeGraph(
        subgraphs={
            "doc-1": {
                "document": [_document_node()],
                "chunks": [
                    _chunk_node("chunk-2", 2, "third"),
                    _chunk_node("chunk-0", 0, "first"),
                    _chunk_node("chunk-1", 1, "second"),
                ],
            }
        }
    )

    result = await get_document_from_graph(
        graph,
        "doc-1",
        include_metadata=True,
        max_chunks=2,
    )

    assert result["document_id"] == "doc-1"
    assert result["chunk_count"] == 2
    assert result["total_chunks"] == 3
    assert result["is_truncated"] is True
    assert [chunk["chunk_id"] for chunk in result["chunks"]] == ["chunk-0", "chunk-1"]
    assert result["metadata"]["raw_data_location"] == "/tmp/guide.txt"


@pytest.mark.asyncio
async def test_get_document_from_graph_accepts_chunk_id_via_connections():
    document = _document_node()
    chunks = [
        _chunk_node("chunk-0", 0, "first"),
        _chunk_node("chunk-1", 1, "second"),
    ]
    edge = {"relationship_name": "is_part_of"}
    graph = FakeGraph(
        nodes={"doc-1": document, "chunk-1": chunks[1]},
        connections={
            "chunk-1": [(chunks[1], edge, document)],
            "doc-1": [(chunks[0], edge, document), (chunks[1], edge, document)],
        },
    )

    result = await get_document_from_graph(graph, "chunk-1", include_metadata=False)

    assert result["document_id"] == "doc-1"
    assert result["name"] == "Guide"
    assert "metadata" not in result
    assert [chunk["chunk_id"] for chunk in result["chunks"]] == ["chunk-0", "chunk-1"]


@pytest.mark.asyncio
async def test_get_chunk_neighbors_from_graph_filters_direction_and_target():
    document = _document_node()
    chunks = [
        _chunk_node("chunk-0", 0, "first"),
        _chunk_node("chunk-1", 1, "second"),
        _chunk_node("chunk-2", 2, "third"),
        _chunk_node("chunk-3", 3, "fourth"),
    ]
    edge = {"relationship_name": "is_part_of"}
    graph = FakeGraph(
        nodes={"doc-1": document, "chunk-1": chunks[1]},
        connections={
            "chunk-1": [(chunks[1], edge, document)],
            "doc-1": [(chunk, edge, document) for chunk in chunks],
        },
    )

    result = await get_chunk_neighbors_from_graph(
        graph,
        "chunk-1",
        neighbor_count=1,
        include_target=False,
        direction="both",
    )

    assert result["document_id"] == "doc-1"
    assert result["target_chunk_index"] == 1
    assert [chunk["chunk_id"] for chunk in result["chunks"]] == ["chunk-0", "chunk-2"]
    assert all(chunk["is_target"] is False for chunk in result["chunks"])


@pytest.mark.asyncio
async def test_get_chunk_neighbors_from_graph_validates_inputs():
    graph = FakeGraph()

    with pytest.raises(ValueError, match="neighbor_count"):
        await get_chunk_neighbors_from_graph(graph, "chunk-1", neighbor_count=11)

    with pytest.raises(ValueError, match="direction"):
        await get_chunk_neighbors_from_graph(graph, "chunk-1", direction="sideways")


@pytest.mark.asyncio
async def test_document_retrieval_tools_format_json(monkeypatch):
    import src.server as server

    class FakeClient:
        async def get_document(self, document_id, include_metadata=True, max_chunks=0):
            return {
                "document_id": document_id,
                "include_metadata": include_metadata,
                "max_chunks": max_chunks,
                "chunks": [],
            }

        async def get_chunk_neighbors(
            self,
            chunk_id,
            neighbor_count=2,
            include_target=True,
            direction="both",
        ):
            return {
                "target_chunk_id": chunk_id,
                "neighbor_count": neighbor_count,
                "include_target": include_target,
                "direction": direction,
                "chunks": [],
            }

    monkeypatch.setattr(server, "cognee_client", FakeClient())

    document_result = await server.get_document("doc-1", include_metadata=False, max_chunks=3)
    neighbors_result = await server.get_chunk_neighbors(
        "chunk-1",
        neighbor_count=1,
        include_target=False,
        direction="forward",
    )

    assert json.loads(document_result[0].text)["document_id"] == "doc-1"
    neighbor_payload = json.loads(neighbors_result[0].text)
    assert neighbor_payload["target_chunk_id"] == "chunk-1"
    assert neighbor_payload["direction"] == "forward"
