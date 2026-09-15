import asyncio
import base64
import hashlib
import importlib
import json
import mimetypes
import os
import re
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
normalize_delete_mode = server_utils.normalize_delete_mode
validate_top_k = server_utils.validate_top_k
get_chunk_neighbors_from_graph = retrieval_utils.get_chunk_neighbors_from_graph
get_document_from_graph = retrieval_utils.get_document_from_graph


def test_validate_top_k_and_delete_mode():
    assert validate_top_k("3") == 3
    assert normalize_delete_mode(" HARD ") == "hard"

    with pytest.raises(ValueError):
        validate_top_k(0)
    with pytest.raises(ValueError):
        validate_top_k(101)
    with pytest.raises(ValueError):
        normalize_delete_mode("unsafe")


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


def test_cognee_client_auth_schemes():
    # 1. Default non-tenant URL -> Bearer token
    client = CogneeClient(api_url="http://localhost:8000", api_token="secret_key")
    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer secret_key"
    assert "X-Api-Key" not in headers

    # 2. Explicit x-api-key scheme -> X-Api-Key header
    client_key = CogneeClient(
        api_url="http://localhost:8000",
        api_token="secret_key",
        api_auth_scheme="x-api-key",
    )
    headers_key = client_key._get_headers()
    assert headers_key["X-Api-Key"] == "secret_key"
    assert "Authorization" not in headers_key

    # 3. Explicit bearer scheme -> Bearer token
    client_bearer = CogneeClient(
        api_url="http://localhost:8000",
        api_token="secret_key",
        api_auth_scheme="bearer",
    )
    headers_bearer = client_bearer._get_headers()
    assert headers_bearer["Authorization"] == "Bearer secret_key"
    assert "X-Api-Key" not in headers_bearer

    # 4. Cloud tenant URL -> X-Api-Key + X-Tenant-Id
    tenant_url = "https://tenant-12345678-1234-1234-1234-123456789abc.cognee.ai"
    client_cloud = CogneeClient(api_url=tenant_url, api_token="secret_key")
    headers_cloud = client_cloud._get_headers()
    assert headers_cloud["X-Api-Key"] == "secret_key"
    assert headers_cloud["X-Tenant-Id"] == "12345678-1234-1234-1234-123456789abc"
    assert "Authorization" not in headers_cloud

    # 5. COGNEE_API_AUTH_SCHEME environment variable
    os.environ["COGNEE_API_AUTH_SCHEME"] = "x-api-key"
    try:
        client_env = CogneeClient(api_url="http://localhost:8000", api_token="secret_key")
        headers_env = client_env._get_headers()
        assert headers_env["X-Api-Key"] == "secret_key"
        assert "Authorization" not in headers_env
    finally:
        os.environ.pop("COGNEE_API_AUTH_SCHEME", None)


# Tools that the MCP server is expected to expose. Kept as named groups so the
# contract documents intent rather than just enumerating names. The hardening
# rule is that the LLM-direct memory API stays minimal (V2: remember/recall/
# forget).
MEMORY_API_TOOLS = {"remember", "recall", "forget"}
STATUS_TOOLS = {
    # Ingestion is queued (remember(background=True)) because it outruns the
    # host's request deadline, so progress and failures are only observable
    # through a status call.
    "cognify_status",
}
EXPECTED_TOOLS = MEMORY_API_TOOLS | STATUS_TOOLS


@pytest.mark.asyncio
async def test_mcp_exposes_only_memory_tools():
    from src import server

    tools = await server.mcp.list_tools()

    assert {tool.name for tool in tools} == EXPECTED_TOOLS


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


async def _mock_api_client(requests: list[httpx.Request], **kwargs) -> CogneeClient:
    """API-mode client whose requests are recorded instead of sent over the wire."""

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = CogneeClient(api_url="http://cognee.local", **kwargs)
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _multipart_filenames(request: httpx.Request) -> list[str]:
    """Filenames declared in a multipart body, in order."""
    body = request.content.decode("latin-1")
    return re.findall(r'filename="([^"]*)"', body)


@pytest.mark.asyncio
async def test_cognee_client_api_add_uploads_file_path_under_original_basename(tmp_path):
    upload = tmp_path / "meeting-notes.md"
    upload.write_text("# Meeting notes\nagenda item", encoding="utf-8")
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests)

    try:
        await client.add(str(upload), dataset_name="ds")
    finally:
        await client.close()

    body = requests[0].content.decode()
    expected_mime = mimetypes.guess_type("meeting-notes.md")[0] or "application/octet-stream"

    assert _multipart_filenames(requests[0]) == ["meeting-notes.md"]
    # Bug from issue 4230: a content-addressed name derived from the path string.
    assert 'filename="text_' not in body
    # The file's bytes must be uploaded, not the path that pointed at them.
    assert "# Meeting notes\nagenda item" in body
    assert str(upload) not in body
    assert f"Content-Type: {expected_mime}" in body
    assert 'name="datasetName"' in body and "ds" in body


@pytest.mark.asyncio
async def test_cognee_client_api_add_accepts_path_objects(tmp_path):
    upload = tmp_path / "client-profile.md"
    upload.write_text("profile", encoding="utf-8")
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests)

    try:
        await client.add(upload, dataset_name="ds")
    finally:
        await client.close()

    assert _multipart_filenames(requests[0]) == ["client-profile.md"]
    assert "profile" in requests[0].content.decode()


@pytest.mark.asyncio
async def test_cognee_client_api_add_uploads_binary_file_bytes(tmp_path):
    upload = tmp_path / "report.pdf"
    raw_bytes = b"%PDF-1.4\n\x00\x80\xff binary payload"
    upload.write_bytes(raw_bytes)
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests)

    try:
        await client.add(str(upload), dataset_name="ds")
    finally:
        await client.close()

    expected_mime = mimetypes.guess_type("report.pdf")[0] or "application/octet-stream"

    assert _multipart_filenames(requests[0]) == ["report.pdf"]
    # Bytes are passed through untouched: a str()/utf-8 round trip would mangle them.
    assert raw_bytes in requests[0].content
    assert f"Content-Type: {expected_mime}".encode() in requests[0].content


@pytest.mark.asyncio
async def test_cognee_client_api_add_keeps_content_addressed_name_for_non_file_text():
    """Text that merely looks like a path stays a content-addressed text upload."""
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests)

    content = "/definitely/not/on/disk/notes.md"
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()

    try:
        await client.add(content, dataset_name="ds")
    finally:
        await client.close()

    body = requests[0].content.decode()
    assert _multipart_filenames(requests[0]) == [f"text_{digest}.txt"]
    assert 'filename="notes.md"' not in body
    assert content in body


@pytest.mark.asyncio
async def test_cognee_client_api_add_keeps_text_upload_for_directories_and_long_text(tmp_path):
    """Non-file arguments must not trip the path branch (or raise from os.path.isfile)."""
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests)

    long_text = "remember this\n" * 5000

    try:
        await client.add(str(tmp_path), dataset_name="ds")
        await client.add(long_text, dataset_name="ds")
    finally:
        await client.close()

    dir_digest = hashlib.md5(str(tmp_path).encode("utf-8")).hexdigest()
    text_digest = hashlib.md5(long_text.encode("utf-8")).hexdigest()

    assert _multipart_filenames(requests[0]) == [f"text_{dir_digest}.txt"]
    assert _multipart_filenames(requests[1]) == [f"text_{text_digest}.txt"]


@pytest.mark.asyncio
async def test_cognee_client_api_remember_uploads_file_path_under_original_basename(tmp_path):
    """remember() shares add()'s path handling (issue #4230, "same pattern")."""
    upload = tmp_path / "alice_notes.md"
    upload.write_text("alice prefers async updates", encoding="utf-8")
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests)

    try:
        await client.remember(str(upload), dataset_name="ds")
    finally:
        await client.close()

    body = requests[0].content.decode()
    assert requests[0].url.path == "/api/v1/remember"
    assert _multipart_filenames(requests[0]) == ["alice_notes.md"]
    assert "alice prefers async updates" in body
    assert str(upload) not in body


@pytest.mark.asyncio
async def test_cognee_client_api_remember_base64_upload_preserves_and_sanitizes_filename():
    """The MCP file-upload path keeps the caller's name but never a directory."""
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests)

    payload = base64.b64encode(b"# Client profile").decode("ascii")

    try:
        await client.remember(
            data=None,
            dataset_name="ds",
            filename="client-profile.md",
            content_base64=payload,
        )
        await client.remember(
            data=None,
            dataset_name="ds",
            filename="../../etc/passwd",
            content_base64=payload,
        )
        await client.remember(data=None, dataset_name="ds", filename="", content_base64=payload)
    finally:
        await client.close()

    assert _multipart_filenames(requests[0]) == ["client-profile.md"]
    assert "# Client profile" in requests[0].content.decode()
    assert _multipart_filenames(requests[1]) == ["passwd.txt"]
    assert _multipart_filenames(requests[2]) == ["upload.txt"]


@pytest.mark.asyncio
async def test_cognee_client_remember_rejects_conflicting_upload_arguments():
    """The client refuses ambiguous payloads before deciding on a transport."""
    client = await _mock_api_client([])
    payload = base64.b64encode(b"file bytes").decode("ascii")

    try:
        with pytest.raises(ValueError, match="not both"):
            await client.remember(data="inline text", filename="notes.md", content_base64=payload)
        with pytest.raises(ValueError, match="do not support session_id"):
            await client.remember(
                data=None, filename="notes.md", content_base64=payload, session_id="s-1"
            )
    finally:
        await client.close()


class FakeCogneeModule:
    """Stand-in for the `cognee` module used by CogneeClient's direct mode."""

    def __init__(self, result=None, error=None):
        self.calls: list[dict] = []
        self.seen_payloads: list[bytes] = []
        self.seen_paths: list[str] = []
        self._result = result
        self._error = error

    async def remember(self, **kwargs):
        self.calls.append(kwargs)
        data = kwargs.get("data")
        # Read through the handed-off path while it is still on disk: the
        # client deletes it as soon as remember() returns.
        if isinstance(data, str) and os.path.isfile(data):
            self.seen_paths.append(data)
            # A blocking read is fine in a test double: no event loop to
            # starve, and the file is a few bytes on tmpfs.
            with open(data, "rb") as handle:
                self.seen_payloads.append(handle.read())
        if self._error is not None:
            raise self._error
        return self._result


def _local_client(fake_cognee: FakeCogneeModule) -> CogneeClient:
    """Direct-mode client with the real cognee module swapped out."""
    client = CogneeClient()
    client.cognee = fake_cognee
    return client


@pytest.mark.asyncio
async def test_cognee_client_local_remember_writes_upload_to_a_temp_file():
    """Direct mode materializes the upload on disk under its original name."""
    fake = FakeCogneeModule()
    client = _local_client(fake)
    raw_bytes = b"%PDF-1.4\n\x00\x80\xff quarterly report"
    payload = base64.b64encode(raw_bytes).decode("ascii")

    result = await client.remember(
        data=None,
        dataset_name="ds",
        filename="q3-report.pdf",
        content_base64=payload,
    )

    assert len(fake.calls) == 1
    handed_off = fake.calls[0]["data"]
    assert Path(handed_off).name == "q3-report.pdf"
    # Bytes survive the base64 -> disk round trip untouched.
    assert fake.seen_payloads == [raw_bytes]
    assert fake.calls[0]["dataset_name"] == "ds"
    # File uploads are permanent-memory only, so no session_id is forwarded.
    assert "session_id" not in fake.calls[0]
    assert result["dataset_name"] == "ds"


@pytest.mark.asyncio
async def test_cognee_client_local_remember_sanitizes_upload_filename():
    """Direct mode applies the same basename sanitizing as the API path."""
    payload = base64.b64encode(b"contents").decode("ascii")

    for supplied, expected in [
        ("../../etc/passwd", "passwd.txt"),
        ("", "upload.txt"),
        ("notes", "notes.txt"),
    ]:
        fake = FakeCogneeModule()
        client = _local_client(fake)

        await client.remember(
            data=None, dataset_name="ds", filename=supplied, content_base64=payload
        )

        written = Path(fake.calls[0]["data"])
        assert written.name == expected
        # The traversal segments must not survive into the staged path.
        assert ".." not in written.parts


@pytest.mark.asyncio
async def test_cognee_client_local_remember_cleans_up_the_temp_upload():
    """The staged file and its directory are removed once ingestion returns."""
    fake = FakeCogneeModule()
    client = _local_client(fake)
    payload = base64.b64encode(b"transient").decode("ascii")

    await client.remember(
        data=None, dataset_name="ds", filename="scratch.md", content_base64=payload
    )

    staged = Path(fake.seen_paths[0])
    assert not staged.exists()
    assert not staged.parent.exists()


@pytest.mark.asyncio
async def test_cognee_client_local_remember_cleans_up_when_ingestion_fails():
    """A failing cognify must not leak the staged upload onto disk."""
    fake = FakeCogneeModule(error=RuntimeError("cognify exploded"))
    client = _local_client(fake)
    payload = base64.b64encode(b"transient").decode("ascii")

    with pytest.raises(RuntimeError, match="cognify exploded"):
        await client.remember(
            data=None, dataset_name="ds", filename="scratch.md", content_base64=payload
        )

    staged = Path(fake.seen_paths[0])
    assert not staged.exists()
    assert not staged.parent.exists()


@pytest.mark.asyncio
async def test_cognee_client_local_remember_passes_text_through_untouched():
    """Text payloads reach cognee.remember as-is, with no file staging."""
    fake = FakeCogneeModule()
    client = _local_client(fake)

    await client.remember(
        data="alice prefers async updates",
        dataset_name="ds",
        session_id="s-1",
        custom_prompt="extract carefully",
    )

    assert fake.calls[0] == {
        "data": "alice prefers async updates",
        "dataset_name": "ds",
        "session_id": "s-1",
        "custom_prompt": "extract carefully",
    }
    assert fake.seen_paths == []


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

    try:
        await client.remember(content, dataset_name="ds", session_id="session-1")
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/remember/entry"
    payload = json.loads(requests[0].content.decode())
    assert payload["dataset_name"] == "ds"
    assert payload["session_id"] == "session-1"
    assert payload["entry"]["type"] == "qa"
    assert payload["entry"]["answer"] == content


@pytest.mark.asyncio
async def test_cognee_client_api_recall_sends_session_id_prompt_and_null_search_type():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.recall(
            "hello",
            session_id="session-1",
            system_prompt="Answer with provenance.",
            top_k=5,
        )
    finally:
        await client.close()

    payload = json.loads(requests[0].content.decode())
    assert requests[0].url.path == "/api/v1/recall"
    assert payload["session_id"] == "session-1"
    assert payload["system_prompt"] == "Answer with provenance."
    assert payload["search_type"] is None
    assert payload["top_k"] == 5


class RecordingRememberClient:
    """Fake cognee_client that records what the remember tool forwarded."""

    def __init__(self):
        self.calls: list[dict] = []

    async def remember(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "completed"}


@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        pytest.param(
            {"data": "inline text", "filename": "n.md", "content_base64": "aGk="},
            "not both",
            id="text_and_file",
        ),
        pytest.param({}, "provide `data`", id="neither"),
        pytest.param({"data": ""}, "provide `data`", id="empty_data"),
        pytest.param(
            {"filename": "n.md", "content_base64": "aGk=", "session_id": "s-1"},
            "don't support session_id",
            id="upload_with_session",
        ),
        pytest.param(
            {"filename": "n.md", "content_base64": "not valid base64!"},
            "invalid base64",
            id="malformed_base64",
        ),
    ],
)
@pytest.mark.asyncio
async def test_mcp_remember_rejects_invalid_payloads(monkeypatch, kwargs, expected_error):
    """Bad argument combinations are refused before any ingestion is attempted."""
    from src import server

    fake_client = RecordingRememberClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    result = await server.remember(**kwargs)

    assert expected_error in result[0].text
    assert result[0].text.startswith("Error:")
    assert fake_client.calls == []


@pytest.mark.asyncio
async def test_mcp_remember_rejects_uploads_over_the_size_limit(monkeypatch):
    """Oversized uploads are rejected client-side rather than posted."""
    from src import server

    fake_client = RecordingRememberClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    oversized = base64.b64encode(b"x" * (server._MAX_UPLOAD_BYTES + 1)).decode("ascii")
    result = await server.remember(filename="big.bin", content_base64=oversized)

    assert "exceeds 10 MB limit" in result[0].text
    assert fake_client.calls == []

    at_limit = base64.b64encode(b"x" * server._MAX_UPLOAD_BYTES).decode("ascii")
    result = await server.remember(filename="big.bin", content_base64=at_limit)

    # The limit itself is inclusive.
    assert not result[0].text.startswith("Error:")
    assert len(fake_client.calls) == 1


@pytest.mark.asyncio
async def test_mcp_remember_forwards_file_uploads(monkeypatch):
    """The merged tool hands filename + content_base64 straight to the client."""
    from src import server

    fake_client = RecordingRememberClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    raw_bytes = b"# Client profile\nprefers async updates"
    payload = base64.b64encode(raw_bytes).decode("ascii")

    result = await server.remember(
        filename="client-profile.md",
        content_base64=payload,
        dataset_name="ds",
        custom_prompt="extract carefully",
    )

    assert fake_client.calls == [
        {
            "data": None,
            "filename": "client-profile.md",
            "content_base64": payload,
            "dataset_name": "ds",
            "session_id": None,
            "custom_prompt": "extract carefully",
        }
    ]
    # The confirmation names the file and its decoded size, not the base64 length.
    assert "client-profile.md" in result[0].text
    assert f"{len(raw_bytes):,} bytes" in result[0].text
    assert "ds" in result[0].text


@pytest.mark.asyncio
async def test_mcp_remember_still_stores_text_and_session_entries(monkeypatch):
    """Absorbing cognify_file left the pre-existing text paths intact."""
    from src import server

    fake_client = RecordingRememberClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    permanent = await server.remember(data="alice prefers async updates", dataset_name="ds")
    session = await server.remember(data="scratch note", dataset_name="ds", session_id="s-1")

    assert fake_client.calls[0]["data"] == "alice prefers async updates"
    assert fake_client.calls[0]["content_base64"] is None
    assert "knowledge graph" in permanent[0].text

    assert fake_client.calls[1]["session_id"] == "s-1"
    assert "session cache" in session[0].text


@pytest.mark.asyncio
async def test_mcp_remember_defaults_dataset_when_caller_omits_it(monkeypatch):
    """Uploads inherit the agent-scoped default dataset, same as text writes."""
    from src import server

    fake_client = RecordingRememberClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "cursor_vscode_memory")

    await server.remember(filename="n.md", content_base64=base64.b64encode(b"hi").decode("ascii"))

    assert fake_client.calls[0]["dataset_name"] == "cursor_vscode_memory"


@pytest.mark.asyncio
async def test_mcp_remember_reports_client_failures(monkeypatch):
    """Ingestion errors surface as tool errors instead of propagating."""
    from src import server

    class ExplodingClient:
        async def remember(self, **kwargs):
            raise RuntimeError("upstream is down")

    monkeypatch.setattr(server, "cognee_client", ExplodingClient())

    result = await server.remember(
        filename="n.md", content_base64=base64.b64encode(b"hi").decode("ascii")
    )

    assert "Remember failed" in result[0].text
    assert "upstream is down" in result[0].text


@pytest.mark.asyncio
async def test_mcp_no_longer_exposes_cognify_file():
    """cognify_file was absorbed into remember and must be gone from the surface."""
    from src import server

    tools = await server.mcp.list_tools()

    assert "cognify_file" not in {tool.name for tool in tools}
    assert not hasattr(server, "cognify_file")


@pytest.mark.asyncio
async def test_mcp_remember_advertises_file_upload_parameters():
    """The merged tool's schema is what tells an LLM it can send files."""
    from src import server

    tools = await server.mcp.list_tools()
    remember_tool = next(tool for tool in tools if tool.name == "remember")
    # FastMCP 3 returns its own Tool objects, whose JSON schema is `parameters`;
    # `inputSchema` is the name on the MCP wire type a client receives.
    properties = remember_tool.parameters["properties"]

    assert {"data", "filename", "content_base64"} <= set(properties)
    assert "filename" in remember_tool.description
    assert "content_base64" in remember_tool.description


@pytest.mark.asyncio
async def test_mcp_recall_forwards_system_prompt(monkeypatch):
    from src import server

    class FakeClient:
        def __init__(self):
            self.recall_kwargs = None

        async def recall(self, **kwargs):
            self.recall_kwargs = kwargs
            return [{"text": "ok"}]

    fake_client = FakeClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    result = await server.recall(
        query="hello",
        search_type="graph_completion",
        datasets="main,docs",
        session_id="session-1",
        system_prompt="Answer with provenance.",
        top_k=5,
    )

    assert "ok" in result[0].text
    assert fake_client.recall_kwargs == {
        "query_text": "hello",
        "search_type": "graph_completion",
        "datasets": ["main", "docs"],
        "session_id": "session-1",
        "system_prompt": "Answer with provenance.",
        "top_k": 5,
    }


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


def _recall_payload(requests: "list[httpx.Request]") -> dict:
    """Payload of the recall POST; a bare recall sends a GET /datasets preflight first."""
    recall_request = next(r for r in requests if r.url.path == "/api/v1/recall")
    return json.loads(recall_request.content.decode())


@pytest.mark.asyncio
async def test_recall_uses_env_default_system_prompt_when_caller_omits_it(monkeypatch):
    """The env default is injected when the caller passes no system_prompt."""
    monkeypatch.setenv("COGNEE_MCP_RECALL_SYSTEM_PROMPT", "Answer with provenance.")
    monkeypatch.delenv("COGNEE_MCP_RECALL_SYSTEM_PROMPT_FILE", raising=False)

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.recall("hello")
    finally:
        await client.close()

    payload = _recall_payload(requests)
    assert payload["system_prompt"] == "Answer with provenance."


@pytest.mark.asyncio
async def test_recall_caller_system_prompt_wins_over_env_default(monkeypatch):
    """An explicit caller prompt always takes precedence over the env default."""
    monkeypatch.setenv("COGNEE_MCP_RECALL_SYSTEM_PROMPT", "Env default.")
    monkeypatch.delenv("COGNEE_MCP_RECALL_SYSTEM_PROMPT_FILE", raising=False)

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.recall("hello", system_prompt="Caller wins.")
    finally:
        await client.close()

    payload = _recall_payload(requests)
    assert payload["system_prompt"] == "Caller wins."


@pytest.mark.asyncio
async def test_recall_reads_env_default_system_prompt_from_file(monkeypatch, tmp_path):
    """The default can be supplied through a file for larger prompts."""
    prompt_file = tmp_path / "recall_prompt.txt"
    prompt_file.write_text("Prompt from file.\n", encoding="utf-8")
    monkeypatch.delenv("COGNEE_MCP_RECALL_SYSTEM_PROMPT", raising=False)
    monkeypatch.setenv("COGNEE_MCP_RECALL_SYSTEM_PROMPT_FILE", str(prompt_file))

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.recall("hello")
    finally:
        await client.close()

    payload = _recall_payload(requests)
    assert payload["system_prompt"] == "Prompt from file."


@pytest.mark.asyncio
async def test_recall_without_env_default_omits_system_prompt(monkeypatch):
    """With no default configured, the payload carries no system_prompt."""
    monkeypatch.delenv("COGNEE_MCP_RECALL_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("COGNEE_MCP_RECALL_SYSTEM_PROMPT_FILE", raising=False)

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.recall("hello")
    finally:
        await client.close()

    payload = _recall_payload(requests)
    assert "system_prompt" not in payload


# ---------------------------------------------------------------------------
# Transport security: Host/Origin (DNS-rebinding) guard
#
# FastMCP installs its guard on the streamable-http app only; create_sse_app()
# accepts no such option, so http_app() silently drops the allow-lists when
# transport="sse" and the guard never runs. The server mounts the middleware
# itself to close that gap. These exercise it over a real ASGI round trip.
#
# A permitted SSE request opens an event stream and never completes, so
# "accepted" is observed as a timeout rather than a status code.
# ---------------------------------------------------------------------------

TRANSPORT_PATHS = {"http": ("/mcp", "POST"), "sse": ("/sse", "GET")}
_ACCEPTED = "accepted"


def _probe(app, transport, path=None, **headers):
    """Return the status code, or _ACCEPTED if the request opened a stream."""
    import logging
    import threading

    from starlette.testclient import TestClient

    default_path, method = TRANSPORT_PATHS[transport]
    target = path or default_path
    result = []

    def run():
        try:
            with TestClient(app, base_url="http://127.0.0.1:8000") as client:
                result.append(client.request(method, target, headers=headers, json={}).status_code)
        except Exception as exc:
            # Broad on purpose. An empty result reads as _ACCEPTED, so a crash
            # in this thread would satisfy the `!= 421` / `!= 404` assertions.
            # Recording it keeps a failure from masquerading as "accepted";
            # the log gives the traceback the assertion message cannot carry.
            logging.getLogger(__name__).exception("probe request failed")
            result.append(repr(exc))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    return result[0] if result else _ACCEPTED


@pytest.mark.parametrize("transport", ["http", "sse"])
@pytest.mark.parametrize("bind", ["0.0.0.0", "127.0.0.1"])
def test_transport_rejects_foreign_host_and_origin(monkeypatch, transport, bind):
    """Both transports reject rebinding attempts, on any bind address.

    DNS rebinding targets loopback services, so 127.0.0.1 must be guarded too.
    """
    monkeypatch.delenv("MCP_DISABLE_DNS_REBINDING_PROTECTION", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)

    from src import server

    app = server._build_http_app(transport, bind)

    assert _probe(app, transport, Host="attacker.example") == 421, (
        f"{transport} on {bind} accepted a foreign Host header"
    )
    assert _probe(app, transport, Origin="http://attacker.example") == 403, (
        f"{transport} on {bind} accepted a foreign Origin header"
    )


@pytest.mark.parametrize("transport", ["http", "sse"])
def test_mcp_allowed_hosts_admits_named_host(monkeypatch, transport):
    """The documented escape hatch works on both transports."""
    monkeypatch.delenv("MCP_DISABLE_DNS_REBINDING_PROTECTION", raising=False)
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "10.0.0.5:*")

    from src import server

    app = server._build_http_app(transport, "0.0.0.0")

    assert _probe(app, transport, Host="10.0.0.5:8000") != 421
    assert _probe(app, transport, Host="attacker.example") == 421


@pytest.mark.parametrize("transport", ["http", "sse"])
def test_dns_rebinding_protection_can_be_disabled(monkeypatch, transport):
    monkeypatch.setenv("MCP_DISABLE_DNS_REBINDING_PROTECTION", "true")
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)

    from src import server

    app = server._build_http_app(transport, "0.0.0.0")

    assert _probe(app, transport, Host="attacker.example") != 421


@pytest.mark.parametrize(
    "transport,explicit,expected",
    [
        ("http", None, "/mcp"),
        ("http", "/custom", "/custom"),
        ("sse", None, "/sse"),
        ("sse", "/events", "/events"),
    ],
)
def test_path_flag_moves_endpoint_without_clobbering_defaults(transport, explicit, expected):
    """--path applies, and omitting it leaves each transport's own default.

    The transports have different defaults (/mcp, /sse). Forwarding a single
    "/mcp" default to both silently relocated the SSE endpoint.
    """
    from src import server

    app = server._build_http_app(transport, "127.0.0.1", explicit)

    assert _probe(app, transport, path=expected) != 404
