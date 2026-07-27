from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.context_global_variables import (
    current_dataset_id,
    embedding_config,
    graph_db_config,
    llm_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.files.storage.config import file_storage_config
from cognee.infrastructure.llm.config import LLMConfig


@pytest.mark.asyncio
async def test_database_context_sets_and_resets_current_dataset_id(monkeypatch):
    dataset_id = uuid4()
    user_id = uuid4()
    current_dataset_id.set("outer")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    async with set_database_global_context_variables(dataset_id, user_id):
        assert current_dataset_id.get() == str(dataset_id)

    assert current_dataset_id.get() == "outer"


@pytest.mark.asyncio
async def test_llm_and_embedding_config_reset_on_exit(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    outer_llm_config = LLMConfig(llm_model="outer-model")
    llm_config.set(outer_llm_config)
    inner_llm_config = LLMConfig(llm_model="inner-model")
    inner_embedding_config = EmbeddingConfig()

    async with set_database_global_context_variables(
        uuid4(),
        uuid4(),
        llm_config=inner_llm_config,
        embedding_config=inner_embedding_config,
    ):
        assert llm_config.get() is inner_llm_config
        assert embedding_config.get() is inner_embedding_config

    assert llm_config.get() is outer_llm_config
    assert embedding_config.get() is None


@pytest.mark.asyncio
async def test_llm_and_embedding_config_reset_on_exception(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    with pytest.raises(RuntimeError):
        async with set_database_global_context_variables(
            uuid4(),
            uuid4(),
            llm_config=LLMConfig(llm_model="inner-model"),
            embedding_config=EmbeddingConfig(),
        ):
            raise RuntimeError("boom")

    assert llm_config.get() is None
    assert embedding_config.get() is None


@pytest.mark.asyncio
async def test_dataset_database_configs_persist_after_exit(monkeypatch):
    """graph/vector/file-storage configs intentionally persist after exit.

    Callers (and integration tests) read the per-dataset databases right after
    a pipeline run, outside the ``async with`` block; only the LLM/embedding
    overrides and the dataset id are restored on exit.
    """
    dataset_id = uuid4()
    user_id = uuid4()
    fake_user = SimpleNamespace(id=user_id, tenant_id=None)
    fake_dataset_database = SimpleNamespace(
        vector_database_provider="lancedb",
        vector_database_url="",
        vector_database_key="",
        vector_database_name="test_vector_db",
        vector_database_connection_info={},
        graph_database_provider="ladybug",
        graph_database_url="",
        graph_database_key="",
        graph_database_name="test_graph_db",
        graph_database_connection_info={},
        graph_dataset_database_handler="ladybug",
    )

    async def fake_get_user(_user_id):
        return fake_user

    async def fake_get_or_create_dataset_database(_dataset, _user):
        return fake_dataset_database

    async def fake_resolve_connection_info(dataset_database):
        return dataset_database

    class FakeDatasetQueue:
        async def ensure_slot(self, dataset):
            pass

        async def release_slot_for(self, dataset):
            pass

    monkeypatch.setattr(
        "cognee.context_global_variables.backend_access_control_enabled", lambda: True
    )
    monkeypatch.setattr("cognee.context_global_variables.get_user", fake_get_user)
    monkeypatch.setattr(
        "cognee.context_global_variables.get_or_create_dataset_database",
        fake_get_or_create_dataset_database,
    )
    monkeypatch.setattr(
        "cognee.context_global_variables.resolve_dataset_database_connection_info",
        fake_resolve_connection_info,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.dataset_queue.dataset_queue", FakeDatasetQueue
    )

    async with set_database_global_context_variables(dataset_id, user_id):
        assert graph_db_config.get()["graph_database_name"] == "test_graph_db"
        assert vector_db_config.get()["vector_db_name"] == "test_vector_db"
        assert file_storage_config.get() is not None

    assert graph_db_config.get()["graph_database_name"] == "test_graph_db"
    assert vector_db_config.get()["vector_db_name"] == "test_vector_db"
    assert file_storage_config.get() is not None
