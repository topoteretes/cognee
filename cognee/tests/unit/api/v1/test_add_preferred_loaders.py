"""Tests for preferred_loaders normalization and options preservation in add() and remember()."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.add import add as add_fn
from cognee.api.v1.add import add as add_pkg
from cognee.api.v1.remember import remember as remember_pkg
from cognee.api.v1.remember.remember import _ADD_ONLY, RememberKwargs, remember
from cognee.api.v1.serve import state as serve_state
from cognee.tasks.ingestion.dlt_config import with_csv_loader_options

add_module = importlib.import_module("cognee.api.v1.add.add")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
add = add_fn


@pytest.fixture
def executor(monkeypatch):
    """add() with the database and pipeline stubbed; the returned mock records the run."""
    dataset = SimpleNamespace(id=uuid4(), name="test_dataset")
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.preflight"),
        "validate_provider_config",
        Mock(),
    )
    monkeypatch.setattr(add_module, "setup", AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )
    monkeypatch.setattr(
        add_module,
        "resolve_authorized_user_dataset",
        AsyncMock(return_value=(user, dataset)),
    )
    monkeypatch.setattr(
        add_module,
        "resolve_dlt_sources",
        AsyncMock(side_effect=lambda data, **kw: (data, None)),
    )
    monkeypatch.setattr(add_module, "refuse_changed_existing_documents", AsyncMock())

    run = AsyncMock(return_value=None)
    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kwargs: run)
    return run


def test_remember_kwargs_annotation_accepts_dict_and_list():
    """RememberKwargs.preferred_loaders must accept both list and dict."""
    annotation = RememberKwargs.__annotations__["preferred_loaders"]
    assert "preferred_loaders" in _ADD_ONLY
    assert annotation in (list | dict, dict | list) or "dict" in str(annotation)


@pytest.mark.asyncio
async def test_add_preserves_dict_preferred_loaders_options(executor):
    """preferred_loaders passed as a dict must retain all options and extraction rules."""
    rules = {
        "title": {"selector": "h1"},
        "links": {"selector": "a", "attr": "href", "all": True},
    }
    input_loaders = {
        "beautiful_soup_loader": {"extraction_rules": rules},
    }

    await add("https://example.com", preferred_loaders=input_loaders)

    call_kwargs = executor.await_args.kwargs
    ingest_task = call_kwargs["tasks"][1]
    # Task(ingest_data, dataset_name, user, node_set, authorized_dataset.id, preferred_loaders, ...)
    resolved_loaders = ingest_task.default_params["args"][4]

    assert isinstance(resolved_loaders, dict)
    assert "beautiful_soup_loader" in resolved_loaders
    assert resolved_loaders["beautiful_soup_loader"]["extraction_rules"] == rules


@pytest.mark.asyncio
async def test_add_normalizes_dict_none_values(executor):
    """preferred_loaders dict with None or empty values must normalize each value to a dict."""
    input_loaders = {
        "csv_loader": None,
        "pdf_loader": {},
    }

    await add("dummy text", preferred_loaders=input_loaders)

    call_kwargs = executor.await_args.kwargs
    ingest_task = call_kwargs["tasks"][1]
    resolved_loaders = ingest_task.default_params["args"][4]

    assert resolved_loaders == {"csv_loader": {}, "pdf_loader": {}}


@pytest.mark.asyncio
async def test_add_preserves_list_of_dicts(executor):
    """preferred_loaders passed as a list of dicts must retain options."""
    rules = {"headings": {"selector": "h2"}}
    input_loaders = [{"beautiful_soup_loader": {"extraction_rules": rules}}]

    await add("https://example.com", preferred_loaders=input_loaders)

    call_kwargs = executor.await_args.kwargs
    ingest_task = call_kwargs["tasks"][1]
    resolved_loaders = ingest_task.default_params["args"][4]

    assert resolved_loaders["beautiful_soup_loader"]["extraction_rules"] == rules


@pytest.mark.asyncio
async def test_add_preserves_mixed_list(executor):
    """preferred_loaders passed as a list of strings and dicts must normalize correctly."""
    rules = {"headings": {"selector": "h2"}}
    input_loaders = ["csv_loader", {"beautiful_soup_loader": {"extraction_rules": rules}}]

    await add("https://example.com", preferred_loaders=input_loaders)

    call_kwargs = executor.await_args.kwargs
    ingest_task = call_kwargs["tasks"][1]
    resolved_loaders = ingest_task.default_params["args"][4]

    assert resolved_loaders["csv_loader"] == {}
    assert resolved_loaders["beautiful_soup_loader"]["extraction_rules"] == rules


@pytest.mark.asyncio
async def test_add_does_not_mutate_caller_dict(executor):
    """add() should not mutate the dictionary passed by the caller."""
    rules = {"title": {"selector": "title"}}
    original = {"beautiful_soup_loader": {"extraction_rules": rules}}
    original_copy = {"beautiful_soup_loader": {"extraction_rules": dict(rules)}}

    await add("https://example.com", preferred_loaders=original)

    assert original == original_copy


def test_with_csv_loader_options_detects_collision_with_dict():
    """Conflicting DLT options between preferred_loaders dict and bare kwargs must raise ValueError."""
    loaders = {"dlt_csv_loader": {"primary_key": "id_from_loader"}}
    conflicting_dlt_opts = {"primary_key": "id_from_kwarg"}

    with pytest.raises(
        ValueError, match=r"DLT option\(s\) given both to add\(\) and under preferred_loaders"
    ):
        with_csv_loader_options(loaders, conflicting_dlt_opts)


def test_with_csv_loader_options_merges_non_conflicting_options():
    """Non-conflicting DLT options must merge cleanly with dict preferred_loaders."""
    loaders = {"dlt_csv_loader": {"write_disposition": "replace"}}
    dlt_opts = {"primary_key": "custom_id"}

    merged = with_csv_loader_options(loaders, dlt_opts)
    assert merged["dlt_csv_loader"]["write_disposition"] == "replace"
    assert merged["dlt_csv_loader"]["primary_key"] == "custom_id"


@pytest.mark.asyncio
async def test_remember_forwards_dict_preferred_loaders_to_add():
    """remember() must forward dictionary preferred_loaders through to add() without losing options."""
    user = MagicMock(id=uuid4())
    dataset = MagicMock(id=uuid4())
    rules = {"title": {"selector": "title"}}
    input_loaders = {"beautiful_soup_loader": {"extraction_rules": rules}}

    with (
        patch("cognee.shared.utils.send_telemetry"),
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch("cognee.modules.engine.operations.setup.setup", AsyncMock()),
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=user)),
        patch.object(
            remember_module,
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(user, [dataset])),
        ),
        patch("cognee.api.v1.add.add", AsyncMock()) as mock_add,
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={})),
    ):
        await remember(
            "https://example.com",
            dataset_name="test_dataset",
            preferred_loaders=input_loaders,
            self_improvement=False,
        )

    assert mock_add.await_args.kwargs["preferred_loaders"] == input_loaders
