"""``strict`` dataset-name resolution (SDK-747).

Name resolution is a membership filter, so a name matching no dataset used to drop
silently out of the scope: a search over ["real", "typo"] answered from "real" alone.
With ``strict=True`` the miss fails the request naming the offending names; the
default stays lenient because write paths rely on an unmatched name meaning "create it".
``search()`` and recall's graph and code lanes are strict; a session-only recall never
resolves names at all.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_ids import get_dataset_ids
from cognee.modules.search.types import SearchType

ids_mod = importlib.import_module("cognee.modules.data.methods.get_dataset_ids")
existing_mod = importlib.import_module(
    "cognee.modules.data.methods.get_authorized_existing_datasets"
)
search_mod = importlib.import_module("cognee.api.v1.search.search")
recall_mod = importlib.import_module("cognee.api.v1.recall.recall")
inner_search_mod = importlib.import_module("cognee.modules.search.methods.search")

TENANT = uuid4()


def dataset(name, tenant_id=TENANT):
    return SimpleNamespace(id=uuid4(), name=name, tenant_id=tenant_id)


@pytest.fixture
def owned(monkeypatch):
    """The datasets ``get_datasets`` reports the user as owning."""
    datasets = []

    async def fake_get_datasets(_user_id):
        return datasets

    monkeypatch.setattr(ids_mod, "get_datasets", fake_get_datasets)
    return datasets


@pytest.fixture
def user():
    return SimpleNamespace(id=uuid4(), tenant_id=TENANT)


@pytest.mark.asyncio
async def test_default_is_lenient_and_drops_unknown_names(owned, user):
    real = dataset("real")
    owned.append(real)

    assert await get_dataset_ids(["real", "typo"], user) == [real.id]


@pytest.mark.asyncio
async def test_strict_raises_naming_only_the_misses(owned, user):
    owned.append(dataset("real"))

    with pytest.raises(DatasetNotFoundError) as error:
        await get_dataset_ids(["real", "typo", "gone", "typo"], user, strict=True)

    assert "'typo'" in error.value.message
    assert "'gone'" in error.value.message
    assert "'real'" not in error.value.message
    assert error.value.message.count("'typo'") == 1
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_strict_treats_another_tenants_dataset_as_missing(owned, user):
    owned.append(dataset("shared", tenant_id=uuid4()))

    with pytest.raises(DatasetNotFoundError, match="'shared'"):
        await get_dataset_ids(["shared"], user, strict=True)


@pytest.mark.asyncio
async def test_strict_passes_when_every_name_resolves(owned, user):
    first, second = dataset("first"), dataset("second")
    owned.extend([first, second])

    assert await get_dataset_ids(["first", "second"], user, strict=True) == [first.id, second.id]


@pytest.mark.asyncio
async def test_strict_ignores_uuid_input(owned, user):
    ids = [uuid4()]

    assert await get_dataset_ids(ids, user, strict=True) == ids


@pytest.mark.asyncio
async def test_authorized_existing_datasets_forwards_strict(monkeypatch, user):
    seen = {}

    async def fake_get_dataset_ids(datasets, _user, strict=False):
        seen["strict"] = strict
        return []

    monkeypatch.setattr(existing_mod, "get_dataset_ids", fake_get_dataset_ids)

    assert await get_authorized_existing_datasets(["x"], "read", user, strict=True) == []
    assert seen["strict"] is True


@pytest.mark.asyncio
async def test_search_rejects_a_mixed_name_list_before_searching(owned, user, monkeypatch):
    """The whole point: a typo must fail the request, not silently shrink the scope."""
    owned.append(dataset("real"))

    async def fake_search_function(**_kwargs):
        raise AssertionError("search ran despite an unresolvable dataset name")

    async def noop(_user):
        return None

    monkeypatch.setattr(search_mod, "search_function", fake_search_function)
    monkeypatch.setattr(search_mod, "set_session_user_context_variable", noop)

    with pytest.raises(DatasetNotFoundError, match="'does_not_exist'"):
        await search_mod.search(
            "what is NLP?",
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=["real", "does_not_exist"],
            user=user,
        )


@pytest.mark.asyncio
async def test_recall_graph_lane_rejects_a_mixed_name_list_before_searching(
    owned, user, monkeypatch
):
    """recall resolves names itself before calling the search; it must be strict too."""
    owned.append(dataset("real"))

    async def fake_authorized_search(**_kwargs):
        raise AssertionError("search ran despite an unresolvable dataset name")

    async def noop(_user):
        return None

    monkeypatch.setattr(inner_search_mod, "authorized_search", fake_authorized_search)
    monkeypatch.setattr(recall_mod, "set_session_user_context_variable", noop)

    with pytest.raises(DatasetNotFoundError, match="'does_not_exist'"):
        await recall_mod.recall(
            "what is NLP?",
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=["real", "does_not_exist"],
            auto_route=False,
            user=user,
        )


@pytest.mark.asyncio
async def test_recall_session_scope_never_resolves_dataset_names(owned, user, monkeypatch):
    """A session-only recall does not touch datasets, so an unknown name is not an error."""
    session_hit = SimpleNamespace(text="from the session")

    async def fake_search_session(**_kwargs):
        return [session_hit]

    async def fail_if_resolved(*_args, **_kwargs):
        raise AssertionError("dataset names were resolved on a session-only recall")

    monkeypatch.setattr(recall_mod, "_search_session", fake_search_session)
    monkeypatch.setattr(recall_mod, "get_authorized_existing_datasets", fail_if_resolved)

    results = await recall_mod.recall(
        "what did we discuss?",
        scope=["session"],
        session_id="s1",
        datasets=["does_not_exist"],
        user=user,
    )

    assert results == [session_hit]
