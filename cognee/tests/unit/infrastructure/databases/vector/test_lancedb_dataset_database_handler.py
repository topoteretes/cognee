from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.infrastructure.databases.dataset_database_handler  # noqa: F401


handler_module = import_module(
    "cognee.infrastructure.databases.vector.lancedb.LanceDBDatasetDatabaseHandler"
)


@pytest.mark.asyncio
async def test_lancedb_dataset_handler_creates_database_parent_directory(tmp_path, monkeypatch):
    """Verify the handler creates the parent directory and returns the LanceDB path."""
    system_root_directory = tmp_path / "system"
    user = SimpleNamespace(id=uuid4())
    dataset_id = uuid4()

    monkeypatch.setattr(
        handler_module,
        "get_base_config",
        lambda: SimpleNamespace(system_root_directory=str(system_root_directory)),
    )
    monkeypatch.setattr(
        handler_module,
        "get_vectordb_config",
        lambda: SimpleNamespace(vector_db_provider="lancedb", vector_db_key=""),
    )

    dataset_config = await handler_module.LanceDBDatasetDatabaseHandler.create_dataset(
        dataset_id, user
    )

    expected_parent = system_root_directory / "databases" / str(user.id)
    assert expected_parent.is_dir()
    assert dataset_config["vector_database_url"] == str(expected_parent / f"{dataset_id}.lance.db")
