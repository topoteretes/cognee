"""A dataset name with spaces or dots is rejected before the dataset is created.

``resolve_authorized_user_datasets`` used to create the dataset first and only
then call ``check_dataset_name``: ``add(..., dataset_name="software engineering")``
persisted the row and raised a bare ``ValueError`` (a 500) afterwards, so the name
showed up in ``/datasets`` while every later call on it failed the same way.
"""

import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.data.exceptions import InvalidDatasetNameError

resolve_mod = importlib.import_module(
    "cognee.modules.pipelines.layers.resolve_authorized_user_datasets"
)
# The package re-exports a function under the submodule's own name, so reach the
# submodule through sys.modules rather than attribute access.
load_mod = sys.modules["cognee.modules.data.methods.load_or_create_datasets"]


@pytest.fixture
def created(monkeypatch):
    """Names the resolve layer went on to look up or create."""
    names = []

    async def fake_existing(datasets, _permission, _user):
        names.extend(datasets)
        return []

    async def fake_create(name, _user):
        names.append(name)
        return SimpleNamespace(id=uuid4(), name=name)

    monkeypatch.setattr(resolve_mod, "get_authorized_existing_datasets", fake_existing)
    monkeypatch.setattr(load_mod, "create_authorized_dataset", fake_create)
    return names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "suggestion"),
    [("software engineering", "software_engineering"), ("notes.v2", "notes_v2")],
)
async def test_invalid_name_is_rejected_and_nothing_is_created(created, name, suggestion):
    user = SimpleNamespace(id=uuid4())

    with pytest.raises(InvalidDatasetNameError) as error:
        await resolve_mod.resolve_authorized_user_datasets(["fine", name], user)

    assert created == []
    assert error.value.status_code == 400
    assert f"'{name}'" in error.value.message
    assert f"'{suggestion}'" in error.value.remediation
    assert isinstance(error.value, ValueError)


@pytest.mark.asyncio
async def test_valid_name_is_still_created(created):
    user = SimpleNamespace(id=uuid4())

    _, datasets = await resolve_mod.resolve_authorized_user_datasets("software_engineering", user)

    assert [d.name for d in datasets] == ["software_engineering"]
