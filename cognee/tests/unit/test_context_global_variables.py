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
        assert current_dataset_id.get() == dataset_id

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

    async def fake_get_dataset_owner_id(_dataset_id):
        return user_id

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
        "cognee.context_global_variables._get_dataset_owner_id", fake_get_dataset_owner_id
    )
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


@pytest.mark.asyncio
async def test_storage_resolves_under_dataset_owner_for_acl_grantee(monkeypatch):
    """Regression for #4829: a non-owner caller must resolve the OWNER's
    storage (paths, registry row), never its own. Without a permission_type,
    no permission check runs here — authorization stays at the API layer."""
    dataset_id = uuid4()
    owner_id = uuid4()
    caller_id = uuid4()
    owner_user = SimpleNamespace(id=owner_id, tenant_id=None)
    fake_dataset_database = SimpleNamespace(
        vector_database_provider="lancedb",
        vector_database_url="",
        vector_database_key="",
        vector_database_name="grantee_vector_db",
        vector_database_connection_info={},
        graph_database_provider="ladybug",
        graph_database_url="",
        graph_database_key="",
        graph_database_name="grantee_graph_db",
        graph_database_connection_info={},
        graph_dataset_database_handler="ladybug",
    )
    get_user_calls = []
    access_checks = []
    registry_users = []

    async def fake_get_user(user_id):
        get_user_calls.append(user_id)
        return owner_user

    async def fake_get_dataset_owner_id(_dataset_id):
        return owner_id

    async def fake_verify_dataset_access(user_id, checked_dataset_id, permission_type):
        access_checks.append((user_id, checked_dataset_id, permission_type))

    async def fake_get_or_create_dataset_database(_dataset, user):
        registry_users.append(user)
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
        "cognee.context_global_variables._get_dataset_owner_id", fake_get_dataset_owner_id
    )
    monkeypatch.setattr(
        "cognee.context_global_variables._verify_dataset_access", fake_verify_dataset_access
    )
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

    async with set_database_global_context_variables(dataset_id, caller_id):
        graph_file_path = graph_db_config.get()["graph_file_path"]

    # No permission_type was passed, so no permission check runs here.
    assert access_checks == []
    # Everything derived from a user must be derived from the OWNER.
    assert get_user_calls == [owner_id]
    assert registry_users == [owner_user]
    assert str(owner_id) in graph_file_path
    assert str(caller_id) not in graph_file_path


@pytest.mark.asyncio
async def test_user_id_is_optional(monkeypatch):
    """Storage resolves from the dataset's owner alone, so entering without a
    user_id works; permission_type without a user_id is refused before any
    storage side effect."""
    from cognee.exceptions import CogneeValidationError

    dataset_id = uuid4()
    owner_id = uuid4()
    owner_user = SimpleNamespace(id=owner_id, tenant_id=None)
    fake_dataset_database = SimpleNamespace(
        vector_database_provider="lancedb",
        vector_database_url="",
        vector_database_key="",
        vector_database_name="ownerless_vector_db",
        vector_database_connection_info={},
        graph_database_provider="ladybug",
        graph_database_url="",
        graph_database_key="",
        graph_database_name="ownerless_graph_db",
        graph_database_connection_info={},
        graph_dataset_database_handler="ladybug",
    )
    get_user_calls = []
    registry_calls = []

    async def fake_get_user(user_id):
        get_user_calls.append(user_id)
        return owner_user

    async def fake_get_dataset_owner_id(_dataset_id):
        return owner_id

    async def fake_get_or_create_dataset_database(_dataset, user):
        registry_calls.append(user)
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
        "cognee.context_global_variables._get_dataset_owner_id", fake_get_dataset_owner_id
    )
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

    async with set_database_global_context_variables(dataset_id):
        assert graph_db_config.get()["graph_database_name"] == "ownerless_graph_db"

    assert get_user_calls == [owner_id]
    assert registry_calls == [owner_user]

    with pytest.raises(CogneeValidationError, match="permission_type requires a user_id"):
        async with set_database_global_context_variables(dataset_id, permission_type="read"):
            pass

    # The refused entry must not have touched storage again.
    assert registry_calls == [owner_user]


@pytest.mark.asyncio
async def test_denied_caller_never_reaches_dataset_storage(monkeypatch):
    """When a permission_type is passed and the caller lacks that grant, entry
    is rejected before any storage side effect (no registry lookup or
    provisioning)."""
    from cognee.modules.users.exceptions import PermissionDeniedError

    dataset_id = uuid4()
    owner_id = uuid4()
    caller_id = uuid4()
    access_checks = []
    registry_calls = []

    async def fake_get_dataset_owner_id(_dataset_id):
        return owner_id

    async def fake_verify_dataset_access(user_id, checked_dataset_id, permission_type):
        access_checks.append((user_id, checked_dataset_id, permission_type))
        raise PermissionDeniedError(f"User {user_id} has no permission on {checked_dataset_id}.")

    async def fake_get_or_create_dataset_database(_dataset, user):
        registry_calls.append(user)

    class FakeDatasetQueue:
        async def ensure_slot(self, dataset):
            pass

        async def release_slot_for(self, dataset):
            pass

    monkeypatch.setattr(
        "cognee.context_global_variables.backend_access_control_enabled", lambda: True
    )
    monkeypatch.setattr(
        "cognee.context_global_variables._get_dataset_owner_id", fake_get_dataset_owner_id
    )
    monkeypatch.setattr(
        "cognee.context_global_variables._verify_dataset_access", fake_verify_dataset_access
    )
    monkeypatch.setattr(
        "cognee.context_global_variables.get_or_create_dataset_database",
        fake_get_or_create_dataset_database,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.dataset_queue.dataset_queue", FakeDatasetQueue
    )

    with pytest.raises(PermissionDeniedError):
        async with set_database_global_context_variables(
            dataset_id, caller_id, permission_type="delete"
        ):
            pass

    assert access_checks == [(caller_id, dataset_id, "delete")]
    assert registry_calls == []


@pytest.mark.asyncio
async def test_dataset_name_is_rejected(monkeypatch):
    """Only a dataset id (UUID or UUID string) may enter the database context.
    Names must be resolved by the caller first; the context variable is left
    untouched when entry is refused."""
    from cognee.exceptions import CogneeValidationError

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    current_dataset_id.set(None)

    with pytest.raises(CogneeValidationError, match="main_dataset"):
        async with set_database_global_context_variables("main_dataset", uuid4()):
            pass

    assert current_dataset_id.get() is None


@pytest.mark.asyncio
async def test_uuid_string_dataset_is_rejected(monkeypatch):
    """One input type: even a valid UUID in string form is refused — callers
    hold real UUID objects, the boundary does no coercion."""
    from cognee.exceptions import CogneeValidationError

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    with pytest.raises(CogneeValidationError, match="must be a dataset id"):
        async with set_database_global_context_variables(str(uuid4()), uuid4()):
            pass
