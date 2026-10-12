"""cognify(data=...) processes only the selected rows of a dataset.

Runs on the default local stack (sqlite) with a mocked LLM, no API keys.
"""

import asyncio
import importlib
import shutil
import tempfile
from pathlib import Path

import pytest

CACHED_CONFIGS = [
    ("cognee.base_config", "get_base_config"),
    ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
    ("cognee.infrastructure.databases.relational.get_relational_engine", "get_relational_engine"),
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
]


def _clear_config_caches():
    for module_name, factory_name in CACHED_CONFIGS:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(scope="module")
def local_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_cognify_selected_test_"))
    import cognee  # (cognee's import runs load_dotenv(override=True))

    mp = pytest.MonkeyPatch()
    for key, value in {
        "DB_PROVIDER": "sqlite",
        "VECTOR_DB_PROVIDER": "lancedb",
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "CACHE_BACKEND": "sqlite",
        "MOCK_EMBEDDING": "true",
        "TELEMETRY_DISABLED": "1",
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    }.items():
        mp.setenv(key, value)
    _clear_config_caches()

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            return KnowledgeGraph(nodes=[], edges=[])
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    yield cognee

    LLMGateway.acreate_structured_output = original
    mp.undo()
    _clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


async def _dataset(name):
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    return next(d for d in await get_datasets(user.id) if d.name == name)


def test_cognify_processes_only_the_selected_rows(local_env):
    cognee = local_env

    async def scenario():
        from cognee.modules.data.methods.get_dataset_data import get_dataset_data

        await cognee.add("Alice maintains the billing service.", dataset_name="selected")
        await cognee.add("Bob owns the deployment pipeline.", dataset_name="selected")
        dataset = await _dataset("selected")
        selected, other = await get_dataset_data(dataset.id)

        await cognee.cognify(datasets=dataset.id, data=[selected])

        status = {row.id: row.pipeline_status for row in await get_dataset_data(dataset.id)}
        assert "cognify_pipeline" in status[selected.id]
        assert "cognify_pipeline" not in status[other.id]

    asyncio.run(scenario())


def test_cognify_rejects_a_row_of_another_dataset(local_env):
    cognee = local_env

    async def scenario():
        from cognee.modules.data.methods.get_dataset_data import get_dataset_data

        await cognee.add("Carol runs the support rota.", dataset_name="foreign")
        await cognee.add("Dave reviews the security audits.", dataset_name="target")
        (foreign,) = await get_dataset_data((await _dataset("foreign")).id)
        target = await _dataset("target")

        with pytest.raises(ValueError, match="do not belong to dataset"):
            await cognee.cognify(datasets=target.id, data=[foreign], raise_on_error=False)

        (row,) = await get_dataset_data((await _dataset("foreign")).id)
        assert "cognify_pipeline" not in (row.pipeline_status or {})

    asyncio.run(scenario())
