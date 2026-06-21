import types
from uuid import uuid4

import pytest

from cognee.modules.data.exceptions import DatasetTypeError


def _make_user(user_id=None, tenant_id=None):
    return types.SimpleNamespace(
        id=user_id or uuid4(),
        tenant_id=tenant_id or uuid4(),
    )


def _make_dataset(*, name="shared_dataset", dataset_id=None, owner_id=None, tenant_id=None):
    return types.SimpleNamespace(
        id=dataset_id or uuid4(),
        name=name,
        owner_id=owner_id or uuid4(),
        tenant_id=tenant_id,
    )


@pytest.fixture
def datasets_mod():
    import importlib

    return importlib.import_module("cognee.modules.data.methods.get_authorized_existing_datasets")


@pytest.mark.asyncio
async def test_resolves_shared_dataset_names_via_acl(monkeypatch, datasets_mod):
    """Non-owner collaborators can resolve dataset names granted via ACL (#2845, #2846)."""
    user = _make_user()
    owner_id = uuid4()
    shared = _make_dataset(name="acme_shared", owner_id=owner_id, tenant_id=user.tenant_id)
    owned = _make_dataset(name="bob_private", owner_id=user.id, tenant_id=user.tenant_id)

    async def fake_get_all_user_permission_datasets(user_arg, permission_type):
        assert user_arg is user
        assert permission_type == "write"
        return [shared, owned]

    monkeypatch.setattr(
        datasets_mod,
        "get_all_user_permission_datasets",
        fake_get_all_user_permission_datasets,
    )

    result = await datasets_mod.get_authorized_existing_datasets(["acme_shared"], "write", user)

    assert result == [shared]


@pytest.mark.asyncio
async def test_resolves_multiple_shared_names(monkeypatch, datasets_mod):
    user = _make_user()
    ds_a = _make_dataset(name="alpha", tenant_id=user.tenant_id)
    ds_b = _make_dataset(name="beta", tenant_id=user.tenant_id)
    ds_other = _make_dataset(name="gamma", tenant_id=user.tenant_id)

    async def fake_get_all_user_permission_datasets(_user, _permission_type):
        return [ds_a, ds_b, ds_other]

    monkeypatch.setattr(
        datasets_mod,
        "get_all_user_permission_datasets",
        fake_get_all_user_permission_datasets,
    )

    result = await datasets_mod.get_authorized_existing_datasets(["beta", "alpha"], "read", user)

    assert result == [ds_a, ds_b]


@pytest.mark.asyncio
async def test_uuid_inputs_delegate_to_specific_permission_lookup(monkeypatch, datasets_mod):
    user = _make_user()
    dataset_id = uuid4()
    expected = [_make_dataset(dataset_id=dataset_id)]

    async def fake_get_specific_user_permission_datasets(user_id, permission_type, dataset_ids):
        assert user_id == user.id
        assert permission_type == "read"
        assert dataset_ids == [dataset_id]
        return expected

    monkeypatch.setattr(
        datasets_mod,
        "get_specific_user_permission_datasets",
        fake_get_specific_user_permission_datasets,
    )

    result = await datasets_mod.get_authorized_existing_datasets([dataset_id], "read", user)

    assert result == expected


@pytest.mark.asyncio
async def test_none_returns_all_permitted_datasets(monkeypatch, datasets_mod):
    user = _make_user()
    expected = [_make_dataset(name="only_one")]

    async def fake_get_all_user_permission_datasets(_user, permission_type):
        assert permission_type == "delete"
        return expected

    monkeypatch.setattr(
        datasets_mod,
        "get_all_user_permission_datasets",
        fake_get_all_user_permission_datasets,
    )

    result = await datasets_mod.get_authorized_existing_datasets(None, "delete", user)

    assert result == expected


@pytest.mark.asyncio
async def test_mixed_name_and_uuid_types_raise(monkeypatch, datasets_mod):
    user = _make_user()

    with pytest.raises(DatasetTypeError):
        await datasets_mod.get_authorized_existing_datasets(["shared", uuid4()], "read", user)


@pytest.mark.asyncio
async def test_denied_dataset_name_raises_permission_error(monkeypatch, datasets_mod):
    """Fail closed when a requested dataset name is not permitted (#2845)."""
    from cognee.modules.users.exceptions import PermissionDeniedError

    user = _make_user()
    allowed = _make_dataset(name="allowed", tenant_id=user.tenant_id)

    async def fake_get_all_user_permission_datasets(_user, _permission_type):
        return [allowed]

    monkeypatch.setattr(
        datasets_mod,
        "get_all_user_permission_datasets",
        fake_get_all_user_permission_datasets,
    )

    with pytest.raises(PermissionDeniedError, match="for all datasets requested"):
        await datasets_mod.get_authorized_existing_datasets(["allowed", "denied"], "read", user)
