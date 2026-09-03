"""Unit tests for get_readable_datasets / get_permitted_dataset_ids.

Consolidated out of duplicated private wrappers (see the review
discussion on cognee/pull/4342) around
get_specific_user_permission_datasets, which raises PermissionDeniedError
when a user simply has zero permitted datasets — an ordinary, common
state that both functions translate to an empty result rather than
propagate. Any other exception is left to propagate.
"""

import sys
from uuid import uuid4

import pytest

import cognee.modules.users.permissions.methods.get_permitted_dataset_ids  # noqa: F401
import cognee.modules.users.permissions.methods.get_readable_datasets  # noqa: F401
from cognee.modules.users.exceptions import PermissionDeniedError

# Both packages' __init__.py do `from .get_x import get_x`, which overwrites the
# `get_x` attribute on the package with the function — so an attribute-style
# import binds to the function, not the module. Go through sys.modules to get
# the actual modules for patching (see test_llm_gateway_usage.py for the same
# pattern against a different package).
readable_module = sys.modules["cognee.modules.users.permissions.methods.get_readable_datasets"]
permitted_ids_module = sys.modules[
    "cognee.modules.users.permissions.methods.get_permitted_dataset_ids"
]


class _FakeDataset:
    def __init__(self, id_):
        self.id = id_


def _user_id():
    return uuid4()


@pytest.mark.asyncio
async def test_get_readable_datasets_returns_datasets(monkeypatch):
    datasets = [_FakeDataset(uuid4()), _FakeDataset(uuid4())]

    async def fake_get_specific(_user_id, _permission_type, _dataset_ids):
        return datasets

    monkeypatch.setattr(readable_module, "get_specific_user_permission_datasets", fake_get_specific)

    result = await readable_module.get_readable_datasets(_user_id())

    assert result == datasets


@pytest.mark.asyncio
async def test_get_readable_datasets_returns_empty_on_permission_denied(monkeypatch):
    async def fake_get_specific(*_args, **_kwargs):
        raise PermissionDeniedError(message="no datasets")

    monkeypatch.setattr(readable_module, "get_specific_user_permission_datasets", fake_get_specific)

    result = await readable_module.get_readable_datasets(_user_id())

    assert result == []


@pytest.mark.asyncio
async def test_get_readable_datasets_propagates_other_errors(monkeypatch):
    async def fake_get_specific(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(readable_module, "get_specific_user_permission_datasets", fake_get_specific)

    with pytest.raises(RuntimeError):
        await readable_module.get_readable_datasets(_user_id())


@pytest.mark.asyncio
async def test_get_permitted_dataset_ids_extracts_ids(monkeypatch):
    id_a, id_b = uuid4(), uuid4()

    async def fake_get_readable_datasets(_user):
        return [_FakeDataset(id_a), _FakeDataset(id_b)]

    monkeypatch.setattr(permitted_ids_module, "get_readable_datasets", fake_get_readable_datasets)

    result = await permitted_ids_module.get_permitted_dataset_ids(_user_id())

    assert result == [id_a, id_b]


@pytest.mark.asyncio
async def test_get_permitted_dataset_ids_empty_when_no_datasets(monkeypatch):
    async def fake_get_readable_datasets(_user):
        return []

    monkeypatch.setattr(permitted_ids_module, "get_readable_datasets", fake_get_readable_datasets)

    result = await permitted_ids_module.get_permitted_dataset_ids(_user_id())

    assert result == []


def test_permission_denied_error_threads_log_kwargs(caplog):
    """The zero-datasets raise is expected and handled — it must be quietable.

    PermissionDeniedError has to accept the base class's log/log_level kwargs
    (COG-6268); without the passthrough, log_level="DEBUG" raised TypeError and
    every zero-dataset lookup emitted a spurious ERROR at construction time.
    """
    import logging

    with caplog.at_level(logging.ERROR):
        PermissionDeniedError(message="no datasets", log_level="DEBUG")

    assert not [r for r in caplog.records if "PermissionDeniedError" in r.getMessage()]
