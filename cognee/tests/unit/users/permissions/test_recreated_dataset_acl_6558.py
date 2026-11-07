import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.permissions.methods import (
    get_specific_user_permission_datasets,
)
from cognee.api.v1.forget.routers.get_forget_router import get_forget_router

_perm_mod = importlib.import_module(
    "cognee.modules.users.permissions.methods.get_specific_user_permission_datasets"
)
_forget_mod = importlib.import_module("cognee.api.v1.forget.forget")
_forget_pkg = importlib.import_module("cognee.api.v1.forget")
_datasets_mod = importlib.import_module("cognee.api.v1.datasets.datasets")


@pytest.mark.asyncio
async def test_denied_existing_dataset_in_same_tenant_includes_missing_acl_diagnostic(monkeypatch):
    """When a dataset was deleted and recreated with the same deterministic ID,
    a previously shared principal lacks an ACL on the recreated Dataset row.
    The 403 PermissionDeniedError must state that the principal has no ACL on
    that dataset and include its created_at timestamp."""
    tenant_id = uuid4()
    shared_user = SimpleNamespace(id=uuid4(), tenant_id=tenant_id)
    recreated_dataset_id = uuid4()
    recreated_at = datetime(2026, 11, 7, 12, 0, 0, tzinfo=timezone.utc)
    recreated_dataset = SimpleNamespace(
        id=recreated_dataset_id,
        name="agent_sessions",
        created_at=recreated_at,
        tenant_id=tenant_id,
    )

    monkeypatch.setattr(_perm_mod, "get_user", AsyncMock(return_value=shared_user))
    monkeypatch.setattr(
        _perm_mod, "get_all_user_permission_datasets", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        _perm_mod,
        "_get_existing_tenant_datasets",
        AsyncMock(return_value=[recreated_dataset]),
    )

    with pytest.raises(PermissionDeniedError) as exc_info:
        await get_specific_user_permission_datasets(
            shared_user.id, "write", [recreated_dataset_id]
        )

    msg = exc_info.value.message
    assert msg.startswith(
        "Request owner does not have necessary permission: [write] for all datasets requested."
    )
    assert f"Principal {shared_user.id} has no [write] ACL on dataset(s):" in msg
    assert str(recreated_dataset_id) in msg
    assert "name='agent_sessions'" in msg
    assert recreated_at.isoformat() in msg


@pytest.mark.asyncio
async def test_denied_nonexistent_or_cross_tenant_dataset_keeps_generic_message(monkeypatch):
    """If the requested dataset ID does not exist in the caller's tenant,
    no dataset metadata is leaked in the 403 message."""
    shared_user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
    unknown_dataset_id = uuid4()

    monkeypatch.setattr(_perm_mod, "get_user", AsyncMock(return_value=shared_user))
    monkeypatch.setattr(
        _perm_mod, "get_all_user_permission_datasets", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        _perm_mod, "_get_existing_tenant_datasets", AsyncMock(return_value=[])
    )

    with pytest.raises(PermissionDeniedError) as exc_info:
        await get_specific_user_permission_datasets(
            shared_user.id, "read", [unknown_dataset_id]
        )

    assert (
        exc_info.value.message
        == "Request owner does not have necessary permission: [read] for all datasets requested."
    )


@pytest.mark.asyncio
async def test_empty_dataset_with_keep_acls_preserves_dataset_and_non_owner_grants():
    """datasets.empty_dataset(..., keep_acls=True) snapshots all ACL grants
    before deleting the dataset's data and restores the Dataset shell and all
    grants (owner + shared principals) afterwards."""
    owner = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
    shared_role_id = uuid4()
    dataset_id = uuid4()
    dataset_obj = SimpleNamespace(
        id=dataset_id,
        name="agent_sessions",
        owner_id=owner.id,
        tenant_id=owner.tenant_id,
    )

    acl_rows = [
        (owner.id, "read"),
        (owner.id, "write"),
        (owner.id, "delete"),
        (owner.id, "share"),
        (shared_role_id, "read"),
        (shared_role_id, "write"),
    ]

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = acl_rows
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    mock_session_cm.__aexit__.return_value = False

    mock_engine = MagicMock()
    mock_engine.get_async_session.return_value = mock_session_cm

    mock_give_perm = AsyncMock()
    mock_get_principal = AsyncMock(side_effect=lambda pid: SimpleNamespace(id=pid))

    mock_ctx_cm = AsyncMock()
    mock_ctx_cm.__aenter__.return_value = None
    mock_ctx_cm.__aexit__.return_value = False

    with (
        patch.object(
            _datasets_mod,
            "set_database_global_context_variables",
            MagicMock(return_value=mock_ctx_cm),
        ),
        patch.object(
            _datasets_mod,
            "get_authorized_dataset",
            AsyncMock(return_value=dataset_obj),
        ),
        patch.object(
            _datasets_mod,
            "delete_dataset_nodes_and_edges",
            AsyncMock(return_value=SimpleNamespace(node_ids=set(), edge_ids=set())),
        ),
        patch.object(
            _datasets_mod,
            "_invalidate_sessions_for_dataset_nonfatal",
            AsyncMock(),
        ),
        patch.object(
            _datasets_mod,
            "_invalidate_sessions_for_deleted_data_nonfatal",
            AsyncMock(),
        ),
        patch(
            "cognee.modules.search.operations.delete_dataset_search_history",
            AsyncMock(),
        ),
        patch(
            "cognee.modules.data.methods.delete_dataset",
            AsyncMock(return_value=None),
        ) as mock_delete_dataset,
        patch(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            return_value=mock_engine,
        ),
        patch(
            "cognee.modules.users.permissions.methods.get_principal",
            mock_get_principal,
        ),
        patch(
            "cognee.modules.users.permissions.methods.give_permission_on_dataset",
            mock_give_perm,
        ),
    ):
        await _datasets_mod.datasets.empty_dataset(
            dataset_id, user=owner, keep_acls=True
        )

    mock_delete_dataset.assert_awaited_once_with(dataset_obj)
    mock_session.add.assert_called_once()
    recreated_ds = mock_session.add.call_args.args[0]
    assert recreated_ds.id == dataset_id
    assert recreated_ds.name == "agent_sessions"
    assert recreated_ds.owner_id == owner.id
    assert recreated_ds.tenant_id == owner.tenant_id

    restored_grants = [
        (call.args[0].id, call.args[1], call.args[2])
        for call in mock_give_perm.await_args_list
    ]
    assert (shared_role_id, dataset_id, "read") in restored_grants
    assert (shared_role_id, dataset_id, "write") in restored_grants
    assert len(restored_grants) == len(acl_rows)


def test_forget_endpoint_passes_keep_acls_flag():
    """POST /api/v1/forget with keepAcls=true passes keep_acls=True to forget()."""
    dataset_id = str(uuid4())
    mock_user = SimpleNamespace(
        id=str(uuid4()),
        email="owner@example.com",
        is_active=True,
        tenant_id=str(uuid4()),
    )

    async def override_user():
        return mock_user

    local_app = FastAPI()
    local_app.include_router(get_forget_router(), prefix="/api/v1/forget")
    local_app.dependency_overrides[get_authenticated_user] = override_user
    with TestClient(local_app) as client:
        with patch.object(_forget_pkg, "forget", new_callable=AsyncMock) as mock_forget:
            mock_forget.return_value = {
                "dataset_id": dataset_id,
                "search_history_removed": True,
                "acls_preserved": True,
                "status": "success",
            }
            response = client.post(
                "/api/v1/forget",
                json={"datasetId": dataset_id, "keepAcls": True},
            )
            assert response.status_code == 200
            assert response.json()["acls_preserved"] is True
            assert mock_forget.call_args.kwargs["keep_acls"] is True

