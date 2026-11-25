"""add() and remember() keep the options given with ``preferred_loaders``.

The dict form ``{"loader": {...options}}`` and the list forms ``["loader"]`` /
``[{"loader": {...}}]`` normalize to the same ``{name: options}`` mapping that
reaches the ingestion task. A dict used to be iterated as its keys, so its options
were silently replaced by ``{}``.
"""

import copy
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.serve import state as serve_state

add_module = importlib.import_module("cognee.api.v1.add.add")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")

_URL = "https://example.com/page"
_RULES = {"links": {"selector": "a", "attr": "href", "all": True}}


@pytest.fixture
def executor(monkeypatch):
    """add() with the database and pipeline stubbed; the returned mock records the run."""
    dataset = SimpleNamespace(id=uuid4(), name="documents")
    user = SimpleNamespace(id=uuid4())
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.preflight"), "validate_provider_config", Mock()
    )
    monkeypatch.setattr(add_module, "setup", AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )
    monkeypatch.setattr(
        add_module, "resolve_authorized_user_dataset", AsyncMock(return_value=(user, dataset))
    )
    monkeypatch.setattr(
        add_module, "resolve_dlt_sources", AsyncMock(side_effect=lambda data, **kw: (data, None))
    )
    monkeypatch.setattr(add_module, "refuse_changed_existing_documents", AsyncMock())
    run = AsyncMock(return_value=None)
    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kwargs: run)
    return run


def _ingested_preferred_loaders(executor):
    """The ``preferred_loaders`` argument handed to the ingest_data task of the last run."""
    tasks = executor.await_args.kwargs["tasks"]
    (ingest,) = [task for task in tasks if task.executable is add_module.ingest_data]
    return ingest.default_params["args"][4]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preferred_loaders, expected",
    [
        (
            {"beautiful_soup_loader": {"extraction_rules": _RULES}},
            {"beautiful_soup_loader": {"extraction_rules": _RULES}},
        ),
        (
            [{"beautiful_soup_loader": {"extraction_rules": _RULES}}],
            {"beautiful_soup_loader": {"extraction_rules": _RULES}},
        ),
        (
            ["text_loader", {"beautiful_soup_loader": {"extraction_rules": _RULES}}, "csv_loader"],
            {
                "text_loader": {},
                "beautiful_soup_loader": {"extraction_rules": _RULES},
                "csv_loader": {},
            },
        ),
        (["text_loader", "csv_loader"], {"text_loader": {}, "csv_loader": {}}),
        ({}, {}),
        ([], {}),
        (None, None),
    ],
    ids=["dict", "list-of-dicts", "mixed-list", "list-of-strings", "empty-dict", "empty", "none"],
)
async def test_preferred_loaders_reach_the_ingest_task_with_their_options(
    executor, preferred_loaders, expected
):
    supplied = copy.deepcopy(preferred_loaders)

    await add_module.add(_URL, preferred_loaders=preferred_loaders)

    ingested = _ingested_preferred_loaders(executor)
    assert ingested == expected
    if expected is not None:
        assert list(ingested) == list(expected)  # loader priority is kept
    assert preferred_loaders == supplied  # the caller's value is not mutated


@pytest.mark.asyncio
async def test_a_dict_form_dlt_loader_entry_keeps_its_options(executor):
    preferred_loaders = {"dlt_csv_loader": {"primary_key": "id"}}
    supplied = copy.deepcopy(preferred_loaders)

    await add_module.add(
        _URL, preferred_loaders=preferred_loaders, dlt_config={"write_disposition": "replace"}
    )

    assert _ingested_preferred_loaders(executor) == {
        "dlt_csv_loader": {"primary_key": "id", "write_disposition": "replace"}
    }
    assert preferred_loaders == supplied

    with pytest.raises(ValueError, match="primary_key"):
        await add_module.add(
            _URL, preferred_loaders=preferred_loaders, dlt_config={"primary_key": "other"}
        )
    assert preferred_loaders == supplied


@pytest.mark.asyncio
async def test_remember_passes_dict_preferred_loaders_to_add_and_not_to_cognify():
    import cognee

    user = MagicMock(id=uuid4())
    dataset = MagicMock(id=uuid4())
    preferred_loaders = {"beautiful_soup_loader": {"extraction_rules": _RULES}}

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
        patch("cognee.api.v1.add.add", AsyncMock()) as add,
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={})) as cognify,
    ):
        await cognee.remember(_URL, preferred_loaders=preferred_loaders, self_improvement=False)

    assert add.await_args.kwargs["preferred_loaders"] == {
        "beautiful_soup_loader": {"extraction_rules": _RULES}
    }
    assert "preferred_loaders" not in cognify.await_args.kwargs
