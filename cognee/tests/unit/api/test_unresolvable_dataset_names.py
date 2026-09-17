"""A dataset name that resolves to nothing fails the request.

Name resolution is a membership filter, so an unknown name used to drop
silently out of the search scope: a recall over ["real", "typo"] answered 200
from "real" alone, with nothing in the response saying "typo" was never
searched. The dataset_ids path has always failed the whole request in that
case; these tests pin the same all-or-nothing contract for names, and that it
is enforced before any search work (including the warm-up probe) happens.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_datasets_by_names
from cognee.modules.search.types import SearchType

names_mod = importlib.import_module("cognee.modules.data.methods.get_authorized_datasets_by_names")
recall_mod = importlib.import_module("cognee.api.v1.recall.recall")
search_mod = importlib.import_module("cognee.api.v1.search.search")
inner_search_mod = importlib.import_module("cognee.modules.search.methods.search")


def dataset(name):
    return SimpleNamespace(id=uuid4(), name=name, owner_id=uuid4(), tenant_id=None)


@pytest.fixture
def existing(monkeypatch):
    """Resolve only the names registered here, like the real membership filter."""
    known = {}

    async def fake_get_authorized_existing_datasets(datasets, _permission_type, _user):
        if not datasets:
            return list(known.values())
        return [known[name] for name in datasets if name in known]

    monkeypatch.setattr(
        names_mod, "get_authorized_existing_datasets", fake_get_authorized_existing_datasets
    )
    return known


@pytest.fixture
def no_search(monkeypatch):
    """Fail loudly if a search is attempted: resolution must reject the request first."""

    async def fake_authorized_search(**_kwargs):
        raise AssertionError("search ran despite an unresolvable dataset name")

    async def fake_set_session_user_context_variable(_user):
        return None

    monkeypatch.setattr(inner_search_mod, "authorized_search", fake_authorized_search)
    monkeypatch.setattr(
        recall_mod, "set_session_user_context_variable", fake_set_session_user_context_variable
    )
    monkeypatch.setattr(
        search_mod, "set_session_user_context_variable", fake_set_session_user_context_variable
    )


@pytest.mark.asyncio
async def test_missing_name_raises_and_names_only_the_misses(existing):
    existing["real"] = dataset("real")
    user = SimpleNamespace(id=uuid4())

    with pytest.raises(DatasetNotFoundError) as error:
        await get_authorized_datasets_by_names(["real", "typo", "gone"], "read", user)

    # The point of failing is telling the caller which names it got wrong.
    assert "'typo'" in error.value.message
    assert "'gone'" in error.value.message
    assert "'real'" not in error.value.message
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_repeated_missing_name_is_reported_once(existing):
    user = SimpleNamespace(id=uuid4())

    with pytest.raises(DatasetNotFoundError) as error:
        await get_authorized_datasets_by_names(["typo", "typo"], "read", user)

    assert error.value.message.count("'typo'") == 1


@pytest.mark.asyncio
async def test_every_name_resolving_returns_them_all(existing):
    existing["first"] = dataset("first")
    existing["second"] = dataset("second")
    user = SimpleNamespace(id=uuid4())

    resolved = await get_authorized_datasets_by_names(["first", "second"], "read", user)

    assert {item.name for item in resolved} == {"first", "second"}


@pytest.mark.asyncio
async def test_empty_list_still_means_every_readable_dataset(existing):
    """`[]` is "all datasets", not "zero names that all failed to resolve"."""
    existing["first"] = dataset("first")
    user = SimpleNamespace(id=uuid4())

    resolved = await get_authorized_datasets_by_names([], "read", user)

    assert [item.name for item in resolved] == ["first"]


@pytest.mark.asyncio
async def test_recall_rejects_a_mixed_name_list_without_searching(existing, no_search):
    existing["real"] = dataset("real")

    with pytest.raises(DatasetNotFoundError) as error:
        await recall_mod.recall(
            "what is NLP?",
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=["real", "does_not_exist"],
            auto_route=False,
            user=SimpleNamespace(id=uuid4()),
        )

    assert "'does_not_exist'" in error.value.message


@pytest.mark.asyncio
async def test_search_rejects_a_mixed_name_list_without_searching(existing, no_search):
    existing["real"] = dataset("real")

    with pytest.raises(DatasetNotFoundError) as error:
        await search_mod.search(
            "what is NLP?",
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=["real", "does_not_exist"],
            user=SimpleNamespace(id=uuid4(), tenant_id=None),
        )

    assert "'does_not_exist'" in error.value.message
