import asyncio
import os
import pathlib

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

import cognee
from cognee.context_global_variables import backend_access_control_enabled
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.search.types import SearchType
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import create_user, get_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.modules.users.roles.methods import add_user_to_role, create_role
from cognee.modules.users.tenants.methods import (
    add_user_to_tenant,
    create_tenant,
    select_tenant,
)

pytestmark = pytest.mark.asyncio


def _extract_dataset_id_from_cognify(cognify_result: dict):
    """Extract dataset_id from cognify output dictionary."""
    for dataset_id, _pipeline_result in cognify_result.items():
        return dataset_id
    return None


async def _reset_engines_and_prune() -> None:
    """Reset db engine caches and prune data/system."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        pass

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.fixture(scope="module")
def event_loop():
    """Single event loop for this module (avoids cross-loop futures)."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest_asyncio.fixture(scope="module")
async def permissions_example_env(tmp_path_factory):
    """One-time environment setup for the permissions example test."""
    # Ensure permissions feature is enabled (example requires it), but don't override if caller set it already.
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "True")

    root = tmp_path_factory.mktemp("permissions_example")
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))

    await _reset_engines_and_prune()
    await engine_setup()

    assert backend_access_control_enabled(), (
        "Expected permissions to be enabled via ENABLE_BACKEND_ACCESS_CONTROL=True"
    )

    yield

    await _reset_engines_and_prune()


async def test_permissions_example_flow(permissions_example_env):
    """Pytest version of `examples/python/permissions_example.py` (same scenarios, asserts instead of prints)."""
    # Patch LLM calls so GRAPH_COMPLETION can run without external API keys.
    llm_patch = patch(
        "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output",
        new_callable=AsyncMock,
        return_value="MOCK_ANSWER",
    )

    # Resolve example data file path (repo-shipped PDF).
    repo_root = pathlib.Path(__file__).resolve().parent
    explanation_file_path = str(repo_root / "test_data" / "artificial-intelligence.pdf")
    assert pathlib.Path(explanation_file_path).exists(), (
        f"Expected example PDF to exist at {explanation_file_path}"
    )

    # Same QUANTUM text as in the example.
    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages
    this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the
    preparation and manipulation of quantum states.
    """

    # Create user_1, add AI dataset.
    user_1 = await create_user("user_1@example.com", "example")
    await cognee.add([explanation_file_path], dataset_name="AI", user=user_1)

    # Create user_2, add QUANTUM dataset.
    user_2 = await create_user("user_2@example.com", "example")
    await cognee.add([text], dataset_name="QUANTUM", user=user_2)

    ai_cognify_result = await cognee.cognify(["AI"], user=user_1)
    quantum_cognify_result = await cognee.cognify(["QUANTUM"], user=user_2)

    ai_dataset_id = _extract_dataset_id_from_cognify(ai_cognify_result)
    quantum_dataset_id = _extract_dataset_id_from_cognify(quantum_cognify_result)
    assert ai_dataset_id is not None
    assert quantum_dataset_id is not None

    with llm_patch:
        # user_1 can read own dataset.
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text="What is in the document?",
            user=user_1,
            datasets=[ai_dataset_id],
        )
    assert isinstance(search_results, list) and len(search_results) == 1
    assert search_results[0]["dataset_name"] == "AI"
    assert search_results[0]["search_result"] == ["MOCK_ANSWER"]

    # user_1 can't read dataset owned by user_2.
    with pytest.raises(PermissionDeniedError):
        await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text="What is in the document?",
            user=user_1,
            datasets=[quantum_dataset_id],
        )

    # user_1 can't add to user_2's dataset.
    with pytest.raises(PermissionDeniedError):
        await cognee.add([explanation_file_path], dataset_id=quantum_dataset_id, user=user_1)

        # user_2 grants read permission to user_1 for QUANTUM dataset.
        await authorized_give_permission_on_datasets(
            user_1.id, [quantum_dataset_id], "read", user_2.id
        )

        with llm_patch:
            # Now user_1 can read QUANTUM dataset via dataset_id.
            search_results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text="What is in the document?",
                user=user_1,
                dataset_ids=[quantum_dataset_id],
            )
        assert isinstance(search_results, list) and len(search_results) == 1
        assert search_results[0]["dataset_name"] == "QUANTUM"
        assert search_results[0]["search_result"] == ["MOCK_ANSWER"]

        # Tenant + role scenario.
        tenant_id = await create_tenant("CogneeLab", user_2.id)
        await select_tenant(user_id=user_2.id, tenant_id=tenant_id)
        role_id = await create_role(role_name="Researcher", owner_id=user_2.id)

        user_3 = await create_user("user_3@example.com", "example")
        await add_user_to_tenant(user_id=user_3.id, tenant_id=tenant_id, owner_id=user_2.id)
        await add_user_to_role(user_id=user_3.id, role_id=role_id, owner_id=user_2.id)
        await select_tenant(user_id=user_3.id, tenant_id=tenant_id)

        # Can't grant role permission on a dataset that isn't part of the active tenant.
        with pytest.raises(PermissionDeniedError):
            await authorized_give_permission_on_datasets(
                role_id, [quantum_dataset_id], "read", user_2.id
            )

        # Re-create QUANTUM dataset in CogneeLab tenant so role permissions can be assigned.
        user_2 = await get_user(user_2.id)  # refresh tenant context
        await cognee.add([text], dataset_name="QUANTUM_COGNEE_LAB", user=user_2)
        quantum_cognee_lab_cognify_result = await cognee.cognify(
            ["QUANTUM_COGNEE_LAB"], user=user_2
        )
        quantum_cognee_lab_dataset_id = _extract_dataset_id_from_cognify(
            quantum_cognee_lab_cognify_result
        )
        assert quantum_cognee_lab_dataset_id is not None

        await authorized_give_permission_on_datasets(
            role_id, [quantum_cognee_lab_dataset_id], "read", user_2.id
        )

        with llm_patch:
            # user_3 can read via role permission.
            search_results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text="What is in the document?",
                user=user_3,
                dataset_ids=[quantum_cognee_lab_dataset_id],
            )
        assert isinstance(search_results, list) and len(search_results) == 1
        assert search_results[0]["dataset_name"] == "QUANTUM_COGNEE_LAB"
        assert search_results[0]["search_result"] == ["MOCK_ANSWER"]
