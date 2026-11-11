"""Who may read the ledger: the one rule the HTTP routes and the CLI share."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.provenance import access
from cognee.modules.provenance.access import (
    authorize_anchoring,
    authorize_ledger_key,
    authorize_ledger_scope,
)
from cognee.modules.provenance.tombstones import ledger_data_item_key
from cognee.modules.users.exceptions import PermissionDeniedError

pytestmark = pytest.mark.asyncio

READABLE, OTHER = uuid4(), uuid4()
USER = SimpleNamespace(id=uuid4(), is_superuser=False)
SUPERUSER = SimpleNamespace(id=uuid4(), is_superuser=True)


@pytest.fixture(autouse=True)
def dataset_acl(monkeypatch):
    async def _authorized(user, dataset_id, permission_type="read"):
        assert permission_type == "read"
        return SimpleNamespace(id=dataset_id) if dataset_id == READABLE else None

    monkeypatch.setattr(access, "get_authorized_dataset", _authorized)


async def test_a_dataset_scope_needs_read_access_to_that_dataset():
    await authorize_ledger_scope(USER, READABLE)
    with pytest.raises(PermissionDeniedError, match=str(OTHER)):
        await authorize_ledger_scope(USER, OTHER)


async def test_the_whole_ledger_is_superuser_only():
    await authorize_ledger_scope(SUPERUSER, None)
    with pytest.raises(PermissionDeniedError, match="superusers only"):
        await authorize_ledger_scope(USER, None)


async def test_a_key_is_read_under_the_dataset_it_names():
    await authorize_ledger_key(USER, ledger_data_item_key(READABLE, uuid4()))
    with pytest.raises(PermissionDeniedError):
        await authorize_ledger_key(USER, ledger_data_item_key(OTHER, uuid4()))


async def test_a_key_that_names_no_dataset_is_superuser_only():
    await authorize_ledger_key(SUPERUSER, "custom-entity")
    with pytest.raises(PermissionDeniedError, match="Unscoped"):
        await authorize_ledger_key(USER, "custom-entity")


async def test_anchoring_is_superuser_only():
    await authorize_anchoring(SUPERUSER)
    with pytest.raises(PermissionDeniedError, match="Anchoring"):
        await authorize_anchoring(USER)
