"""The full rebuild never loses the document.

The rebuild used to delete the row and re-add it pinned to the same id, so a
re-add that failed left no document at all. Now the document's memory is
dropped while the row and its stored files stay, and the pinned re-add
refreshes the row in place. A re-add that fails leaves a document with no
graph that the next cognify() rebuilds from its stored content.

Default local stack in a scratch root, mocked LLM and embeddings.
"""

import re
import shutil
import tempfile
from pathlib import Path

import pytest

from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def durability_env():
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_update_durability_"))
    import cognee

    os.environ.update(
        **incremental_test_backend_env(),
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL=os.environ.get("INCR_TEST_ACL", "true"),
    )
    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
        ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
        (
            "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
            "create_embedding_engine",
        ),
        ("cognee.infrastructure.llm.config", "get_llm_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(MARKER.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=f"marker {n}") for n in names],
                edges=[],
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        return response_model() if isinstance(response_model, type) else "mock"

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate
    yield root
    LLMGateway.acreate_structured_output = original
    shutil.rmtree(root, ignore_errors=True)


async def _entity_names(dataset_id, user_id):
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    async with set_database_global_context_variables(dataset_id, user_id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
    return {
        str(props.get("name", "")).lower() for _, props in nodes if props.get("type") == "Entity"
    }


@pytest.mark.asyncio
async def test_a_failed_rebuild_keeps_the_document_and_the_next_cognify_restores_it(
    durability_env, monkeypatch
):
    await reset_backend_state()
    import sys

    import cognee
    import cognee.api.v1.update.update  # bind the real submodule

    # The package re-exports the update() function under the module's name, so a
    # dotted import yields the function; take the module for patching.
    update_module = sys.modules["cognee.api.v1.update.update"]
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    report = durability_env / "report.txt"
    report.write_text("Report ENTALPHA, version one.\n")
    await cognee.add(str(report), dataset_name="durable")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "durable")
    await cognee.cognify(datasets=[dataset.id])
    (row,) = await get_dataset_data(dataset.id)
    assert "entalpha" in await _entity_names(dataset.id, user.id)

    # The re-add blows up mid-rebuild.
    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure inside the re-add")

    monkeypatch.setattr(update_module, "add", _boom)
    report.write_text("Report ENTBETA, version two.\n")
    with pytest.raises(RuntimeError, match="simulated failure"):
        await cognee.update(row.id, str(report), dataset.id, user=user, chunk_level_diff=False)
    monkeypatch.undo()

    # The document is still there under its id, with its stored content, and
    # its memory is gone rather than half-replaced.
    (survivor,) = await get_dataset_data(dataset.id)
    assert survivor.id == row.id
    assert survivor.content_hash == row.content_hash, "a failed rebuild leaves the old content"
    assert not await _entity_names(dataset.id, user.id)

    # cognify() rebuilds the old content from the row; nothing was lost.
    await cognee.cognify(datasets=[dataset.id])
    assert "entalpha" in await _entity_names(dataset.id, user.id)

    # And the update itself works once the cause is gone: same id, new content.
    result = await cognee.update(row.id, str(report), dataset.id, user=user, chunk_level_diff=False)
    assert result["status"] == "full_rebuild" and result["data_id"] == row.id
    (updated,) = await get_dataset_data(dataset.id)
    assert updated.id == row.id and updated.content_hash != row.content_hash
    names = await _entity_names(dataset.id, user.id)
    assert "entbeta" in names and "entalpha" not in names
