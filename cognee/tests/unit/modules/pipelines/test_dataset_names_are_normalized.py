"""Dataset names with spaces or dots are normalized, not rejected.

``resolve_authorized_user_datasets`` used to create the dataset first and only
then call ``check_dataset_name``: ``add(..., dataset_name="software engineering")``
persisted the row and raised afterwards, so the name showed up in ``/datasets``
while every later call on it (cognify, search, another add) failed the same way.
Now every run of whitespace or dots becomes one underscore before lookup and
creation, on the write path and on the read-by-name path alike.
"""

import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.data.methods import normalize_dataset_name
from cognee.modules.data.methods.get_dataset_ids import get_dataset_ids

resolve_mod = importlib.import_module(
    "cognee.modules.pipelines.layers.resolve_authorized_user_datasets"
)
ids_mod = importlib.import_module("cognee.modules.data.methods.get_dataset_ids")
# The package re-exports a function under the submodule's own name, so reach the
# submodule through sys.modules rather than attribute access.
load_mod = sys.modules["cognee.modules.data.methods.load_or_create_datasets"]

TENANT = uuid4()


@pytest.fixture
def user():
    return SimpleNamespace(id=uuid4(), tenant_id=TENANT)


@pytest.fixture
def owned(monkeypatch):
    """Datasets the user already owns, as seen by both lookup paths."""
    datasets = []

    async def fake_get_datasets(_user_id):
        return datasets

    monkeypatch.setattr(ids_mod, "get_datasets", fake_get_datasets)
    return datasets


@pytest.fixture
def created(monkeypatch):
    """Names the resolve layer went on to create."""
    names = []

    async def fake_create(name, _user):
        names.append(name)
        return SimpleNamespace(id=uuid4(), name=name, tenant_id=TENANT)

    monkeypatch.setattr(load_mod, "create_authorized_dataset", fake_create)
    return names


@pytest.mark.parametrize(
    ("raw", "stored"),
    [
        ("software engineering", "software_engineering"),
        ("software   engineering", "software_engineering"),
        ("notes.v2", "notes_v2"),
        (" padded name ", "padded_name"),
        ("already_fine", "already_fine"),
    ],
)
def test_normalize_dataset_name(raw, stored):
    assert normalize_dataset_name(raw) == stored


@pytest.mark.asyncio
async def test_resolve_creates_the_normalized_name(owned, created, user):
    _, datasets = await resolve_mod.resolve_authorized_user_datasets("software engineering", user)

    assert created == ["software_engineering"]
    assert [d.name for d in datasets] == ["software_engineering"]


@pytest.mark.asyncio
async def test_resolve_reuses_the_existing_dataset_under_either_spelling(
    owned, created, user, monkeypatch
):
    existing = SimpleNamespace(id=uuid4(), name="software_engineering", tenant_id=TENANT)
    owned.append(existing)

    async def fake_existing(datasets, _permission, _user):
        return [d for d in owned if d.name in datasets]

    monkeypatch.setattr(resolve_mod, "get_authorized_existing_datasets", fake_existing)
    _, datasets = await resolve_mod.resolve_authorized_user_datasets("software engineering", user)

    assert created == []
    assert datasets == [existing]


@pytest.mark.asyncio
async def test_read_path_resolves_a_spaced_name_to_the_stored_dataset(owned, user):
    stored = SimpleNamespace(id=uuid4(), name="software_engineering", tenant_id=TENANT)
    owned.append(stored)

    assert await get_dataset_ids(["software engineering"], user, strict=True) == [stored.id]
