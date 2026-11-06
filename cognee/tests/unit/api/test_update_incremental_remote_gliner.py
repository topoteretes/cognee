"""Chunk-level update() honours the remote GLiNER worker (SDK-980).

``ensure_extractor_runtime`` installs no local GLiNER runtime when
``COGNEE_GLINER_TRANSPORT`` names a worker, so the incremental extraction step
must send its model calls there too, as cognify does.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.update import incremental
from cognee.modules.cognify.config import GLINER_DEMO_EXTRACTOR
from cognee.modules.data.processing.document_types import TextDocument
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.graph.gliner_demo import tasks as gliner_tasks
from cognee.tasks.graph.gliner_demo.remote import RemoteGlinerAdapter


@pytest.fixture
def gliner_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("COGNEE_GLINER_"):
            monkeypatch.delenv(name)
    return monkeypatch


async def _run_extraction_step():
    document = TextDocument(name="doc.txt", raw_data_location="doc.txt", external_metadata=None)
    prepare = AsyncMock(return_value=[document])
    extract = AsyncMock(return_value=[])
    with (
        patch.object(gliner_tasks, "prepare_gliner_schema", prepare),
        patch.object(gliner_tasks, "extract_graph_and_summarize_with_gliner", extract),
        patch.object(incremental, "get_max_chunk_tokens", AsyncMock(return_value=512)),
        patch.object(incremental, "_resolve_extraction_config", return_value={}),
    ):
        step = await incremental._extraction_step(
            GLINER_DEMO_EXTRACTOR,
            document,
            incremental.TextChunker,
            KnowledgeGraph,
            None,
            SimpleNamespace(summary_method=None),
        )
        await step(["chunk"], None)
    return prepare, extract


@pytest.mark.asyncio
async def test_a_configured_worker_runs_both_model_calls_of_an_update(gliner_env):
    gliner_env.setenv("COGNEE_GLINER_TRANSPORT", "http")
    gliner_env.setenv("COGNEE_GLINER_ENDPOINT", "http://worker:8080")

    with patch.object(RemoteGlinerAdapter, "ensure_ready", AsyncMock()) as ensure_ready:
        prepare, extract = await _run_extraction_step()

    ensure_ready.assert_awaited_once()
    remote = prepare.await_args.kwargs["remote"]
    assert isinstance(remote, RemoteGlinerAdapter)
    # The probe and the extraction share one adapter, as in cognify.
    assert extract.await_args.kwargs["remote"] is remote


@pytest.mark.asyncio
async def test_without_a_worker_an_update_extracts_locally(gliner_env):
    prepare, extract = await _run_extraction_step()

    assert prepare.await_args.kwargs["remote"] is None
    assert extract.await_args.kwargs["remote"] is None
