"""CI-safe smoke test for examples/demos/google_drive_ingestion_example.py.

Runs the example's ``main()`` end-to-end with the Drive API mocked and the
LLM mocked (reusing the ``LLMGateway.acreate_structured_output`` monkeypatch
pattern from ``test_cognify_rollback_recovery.py``), plus embedding
indexing no-op'd the same way that test does. No live Google or LLM
credentials are used or required.
"""

import importlib.util
import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.tasks.ingestion.connectors import google_drive as gd_source

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DOC_MIME = "application/vnd.google-apps.document"
EXAMPLE_PATH = (
    pathlib.Path(__file__).parents[4] / "examples" / "demos" / "google_drive_ingestion_example.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location("google_drive_ingestion_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Req:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeFiles:
    def __init__(self, service):
        self._service = service

    def list(self, q, fields, pageSize, pageToken=None):
        import re

        folder_id = re.search(r"'([^']+)' in parents", q).group(1)
        if "mimeType=" in q:
            return _Req({"files": []})
        files = [m for m in self._service.files_by_folder.get(folder_id, []) if not m["trashed"]]
        return _Req({"files": files})

    def get(self, fileId, fields):
        return _Req(self._service.file_by_id[fileId])


class _FakeChanges:
    def __init__(self, service):
        self._service = service

    def getStartPageToken(self):
        return _Req({"startPageToken": self._service.start_token})

    def list(self, pageToken, fields):
        return _Req(
            self._service.changes_by_token.get(
                pageToken, {"changes": [], "newStartPageToken": pageToken}
            )
        )


class _FakeDriveService:
    def __init__(self, files_by_folder, file_by_id, start_token="t0"):
        self.files_by_folder = files_by_folder
        self.file_by_id = file_by_id
        self.start_token = start_token
        self.changes_by_token = {}

    def files(self):
        return _FakeFiles(self)

    def changes(self):
        return _FakeChanges(self)


def _file_meta(file_id):
    return {
        "id": file_id,
        "name": file_id,
        "mimeType": DOC_MIME,
        "parents": ["root"],
        "trashed": False,
        "webViewLink": f"https://drive/{file_id}",
        "modifiedTime": "2026-01-01T00:00:00Z",
        "size": None,
    }


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    from cognee.shared.data_models import KnowledgeGraph, Node as KGNode, SummarizedContent

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        return KnowledgeGraph(
            nodes=[KGNode(id="Acme", name="Acme", type="Company", description="A company")],
            edges=[],
        )
    return response_model()


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "root")
    monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "service_account")
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "unused.json")

    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    async def _noop_index(*_args, **_kwargs):
        return None

    monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)

    async def _mock_embed_text(self, text):
        return [[0.0] * self.get_vector_size() for _ in text]

    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _mock_embed_text)

    file_a = _file_meta("fileA")
    service = _FakeDriveService(files_by_folder={"root": [file_a]}, file_by_id={"fileA": file_a})
    monkeypatch.setattr(gd_source, "build_drive_service", lambda **kwargs: service)
    monkeypatch.setattr(
        gd_source,
        "extract_file_content",
        lambda service, file_id, mime_type, name: "Acme is a company based in Springfield.",
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.mark.asyncio
async def test_example_runs_without_live_credentials(clean_environment):
    pytest.importorskip("ladybug")

    module = _load_example_module()
    await module.main()
