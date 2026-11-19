import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)

_lookup = importlib.import_module(get_specific_user_permission_datasets.__module__)


def _patch_lookup(monkeypatch, datasets):
    user_id = uuid4()

    async def fake_get_user(_user_id):
        return SimpleNamespace(id=user_id)

    async def fake_datasets(_user, _permission_type):
        return datasets

    monkeypatch.setattr(_lookup, "get_user", fake_get_user)
    monkeypatch.setattr(_lookup, "get_all_user_permission_datasets", fake_datasets)
    return user_id


@pytest.mark.asyncio
async def test_missing_grant_on_named_datasets_says_to_grant_again(monkeypatch):
    user_id = _patch_lookup(monkeypatch, [])

    with pytest.raises(PermissionDeniedError) as caught:
        await get_specific_user_permission_datasets(user_id, "write", [uuid4()])

    error = caught.value
    assert "necessary permission: [write]" in error.message
    assert "not restored when the same id is created again" in error.message
    assert "created_at" in error.remediation
    assert "Grant the permission again" in error.remediation


@pytest.mark.asyncio
async def test_partial_grant_uses_the_same_denial(monkeypatch):
    allowed = SimpleNamespace(id=uuid4())
    user_id = _patch_lookup(monkeypatch, [allowed])

    with pytest.raises(PermissionDeniedError) as caught:
        await get_specific_user_permission_datasets(user_id, "read", [allowed.id, uuid4()])

    assert "necessary permission: [read]" in caught.value.message
    assert caught.value.remediation


@pytest.mark.asyncio
async def test_no_datasets_at_all_stays_the_empty_case(monkeypatch):
    user_id = _patch_lookup(monkeypatch, [])

    with pytest.raises(PermissionDeniedError) as caught:
        await get_specific_user_permission_datasets(user_id, "read")

    assert caught.value.message == "Request owner does not have permission: [read] for any dataset."
    assert caught.value.remediation is None
