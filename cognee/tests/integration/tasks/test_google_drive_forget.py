"""Forget-on-delete must remove a file's cognified graph content, not just its
Data record. Ingests two Drive files, cognifies (LLM + embeddings mocked, no
live creds), deletes one file, and asserts the deleted file's extracted entity
is gone from the graph while the surviving file's entity remains.

Regression test for the orphan-cleanup gap where _delete_dlt_orphans skipped
graph/vector deletion on graph-provenance graphs (has_data_related_nodes only
checks the relational ledger), leaving a forgotten file still retrievable.
"""

import importlib

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.tasks.ingestion.connectors import google_drive_source

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "gdrive_forget_test"
DOC = "application/vnd.google-apps.document"
# Distinctive, unique tokens so each file maps to exactly one graph entity.
ALPHA = "Alphacorp"
BRAVO = "Bravocorp"


class _Req:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _Files:
    def __init__(self, svc):
        self.svc = svc

    def list(self, q, fields, pageSize, pageToken=None):
        import re

        folder = re.search(r"'([^']+)' in parents", q).group(1)
        if "mimeType=" in q:
            return _Req({"files": []})
        return _Req({"files": [m for m in self.svc.by_folder.get(folder, []) if not m["trashed"]]})

    def get(self, fileId, fields):
        return _Req(self.svc.by_id[fileId])

    def export(self, fileId, mimeType):
        return _Req(self.svc.content[fileId].encode("utf-8"))

    def get_media(self, fileId):
        return _Req(self.svc.content[fileId].encode("utf-8"))


class _Changes:
    def __init__(self, svc):
        self.svc = svc

    def getStartPageToken(self):
        return _Req({"startPageToken": self.svc.token})

    def list(self, pageToken, fields):
        return _Req(
            self.svc.changefeed.get(pageToken, {"changes": [], "newStartPageToken": pageToken})
        )


class FakeDrive:
    def __init__(self):
        self.by_folder = {"root": []}
        self.by_id = {}
        self.content = {}
        self.token = "t0"
        self.changefeed = {}

    def files(self):
        return _Files(self)

    def changes(self):
        return _Changes(self)

    def put(self, fid, content):
        meta = {
            "id": fid,
            "name": fid,
            "mimeType": DOC,
            "parents": ["root"],
            "trashed": False,
            "webViewLink": f"https://drive/{fid}",
            "modifiedTime": "2026-01-01T00:00:00Z",
            "size": None,
        }
        self.by_id[fid] = meta
        self.content[fid] = content
        self.by_folder["root"] = [m for m in self.by_folder["root"] if m["id"] != fid] + [meta]

    def remove(self, fid):
        self.by_id.pop(fid, None)
        self.content.pop(fid, None)
        self.by_folder["root"] = [m for m in self.by_folder["root"] if m["id"] != fid]


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    """Extract one entity named after whichever token appears in the chunk text."""
    from cognee.shared.data_models import KnowledgeGraph, Node as KGNode, SummarizedContent

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        name = next((t for t in (ALPHA, BRAVO) if text_input and t in text_input), None)
        nodes = (
            [KGNode(id=name, name=name, type="Company", description=f"{name} entity")]
            if name
            else []
        )
        return KnowledgeGraph(nodes=nodes, edges=[])
    return response_model()


async def _graph_has(token: str) -> bool:
    nodes, _ = await (await get_graph_engine()).get_graph_data()
    token = token.lower()
    for _nid, props in nodes:
        if any(token in str(v).lower() for v in (props or {}).values()):
            return True
    return False


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    async def _noop_index(*_a, **_k):
        return None

    monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)

    async def _mock_embed_text(self, text):
        return [[0.0] * self.get_vector_size() for _ in text]

    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _mock_embed_text)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _sync(drive):
    await cognee.add(
        google_drive_source(folder_id="root", service=drive),
        dataset_name=DATASET,
        primary_key="file_id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    await cognee.cognify(datasets=[DATASET])


@pytest.mark.asyncio
async def test_deleting_a_file_forgets_its_graph_content(clean_environment):
    drive = FakeDrive()
    drive.put("fileA", f"{ALPHA} is a company in the logistics sector.")
    drive.put("fileB", f"{BRAVO} is an unrelated company in the finance sector.")

    await _sync(drive)
    assert await _graph_has(ALPHA), "fileA entity should be in the graph after ingest"
    assert await _graph_has(BRAVO), "fileB entity should be in the graph after ingest"

    # Delete fileB from Drive; the incremental re-sync must forget its content.
    drive.remove("fileB")
    drive.changefeed["t0"] = {
        "changes": [{"fileId": "fileB", "removed": True}],
        "newStartPageToken": "t1",
    }
    await _sync(drive)

    assert await _graph_has(ALPHA), "surviving fileA entity must remain after deleting fileB"
    assert not await _graph_has(BRAVO), (
        "deleted fileB's entity must be removed from the graph (forget-on-delete), "
        "not just its Data record"
    )
