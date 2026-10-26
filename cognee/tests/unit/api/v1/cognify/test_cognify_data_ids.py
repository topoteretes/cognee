"""``cognify(data_ids=...)`` builds only those items, in a run of their own.

GitHub sync builds each repository this way, so a repository whose
build fails rolls back its own ``cognify_pipeline`` run only, and an earlier
repository still pending in the dataset is not rebuilt by every later one.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.serve import state as serve_state
from cognee.exceptions import CogneeValidationError

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
migrations = importlib.import_module("cognee.modules.migrations.startup")

DATASET = SimpleNamespace(id=uuid4(), name="repos")


def _item(source="code_repo"):
    return SimpleNamespace(
        id=uuid4(), extension="json", system_metadata={"source": source}, pipeline_status={}
    )


REPO_A = _item()
REPO_B = _item()
DOCUMENT = _item(source=None)


async def _fake_dataset_data(dataset_id, data_ids=None, **_kwargs):
    rows = [REPO_A, REPO_B, DOCUMENT]
    return rows if data_ids is None else [row for row in rows if row.id in data_ids]


async def _cognify(datasets=("repos",), authorized=None, **cognify_kwargs):
    async def execute(**kwargs):
        if "before_run" in kwargs:
            await kwargs["before_run"](kwargs["data"], DATASET)
        return {DATASET.id: SimpleNamespace(status="done")}

    pipeline = AsyncMock(side_effect=execute)
    runtime = AsyncMock()
    if authorized is None:
        authorized = AsyncMock(return_value=[DATASET])
    dataset_data = AsyncMock(side_effect=_fake_dataset_data)

    with (
        patch.object(cognify_module, "ensure_extractor_runtime", runtime),
        patch.object(migrations, "run_migrations_and_block", AsyncMock()),
        patch(
            "cognee.modules.users.methods.get_default_user",
            AsyncMock(return_value=SimpleNamespace(id=uuid4())),
        ),
        patch("cognee.modules.data.methods.get_authorized_existing_datasets", authorized),
        patch("cognee.modules.data.methods.get_dataset_data", dataset_data),
        patch.object(cognify_module, "get_pipeline_executor", return_value=pipeline),
        patch(
            "cognee.modules.pipelines.layers.setup_and_check_environment.setup_and_check_environment",
            AsyncMock(),
        ),
    ):
        result = await cognify_module.cognify(datasets=list(datasets), **cognify_kwargs)
    return result, pipeline, runtime, dataset_data


@pytest.mark.asyncio
async def test_data_ids_scope_the_run_to_those_rows():
    _result, pipeline, runtime, dataset_data = await _cognify(data_ids=[REPO_B.id])

    pipeline.assert_awaited_once()
    kwargs = pipeline.await_args.kwargs
    assert kwargs["data"] == [REPO_B]
    # Still the cognify pipeline (status, improve's write gate, the rollback),
    # still under the dataset's lock and permission checks in run_pipeline.
    assert kwargs["pipeline_name"] == "cognify_pipeline"
    assert kwargs["datasets"] == ["repos"]
    assert kwargs["rollback_handler"] is cognify_module.cognify_rollback_handler
    # The pending document outside the scope does not pull in the extractor.
    runtime.assert_not_awaited()
    # Only the requested rows are read, never the whole dataset.
    assert [call.kwargs.get("data_ids") for call in dataset_data.await_args_list] == [{REPO_B.id}]


@pytest.mark.asyncio
async def test_ids_outside_the_dataset_run_nothing():
    # A selection is never narrowed silently, and an empty one never widens to the dataset.
    foreign = uuid4()
    with pytest.raises(CogneeValidationError, match=f"not data items.*{foreign}"):
        await _cognify(data_ids=[REPO_A.id, foreign])
    with pytest.raises(CogneeValidationError, match="data_ids is empty"):
        await _cognify(data_ids=[])


@pytest.mark.parametrize(
    "cognify_kwargs, message",
    [
        ({"datasets": ["a", "b"]}, "exactly one dataset"),
        ({"dry_run": True}, "data_ids is not supported with dry_run"),
    ],
    ids=["several_datasets", "dry_run"],
)
@pytest.mark.asyncio
async def test_data_ids_are_refused(cognify_kwargs, message):
    with pytest.raises(CogneeValidationError, match=message):
        await _cognify(data_ids=[REPO_A.id], **cognify_kwargs)


@pytest.mark.asyncio
async def test_a_misspelled_dataset_name_raises():
    from cognee.modules.data.exceptions import DatasetNotFoundError

    async def strict_lookup(datasets, permission_type, user, strict=False):
        assert (permission_type, strict) == ("write", True)
        raise DatasetNotFoundError(message="Dataset 'repso' not found.")

    with pytest.raises(DatasetNotFoundError):
        await _cognify(
            datasets=["repso"],
            authorized=AsyncMock(side_effect=strict_lookup),
            data_ids=[REPO_A.id],
        )


@pytest.mark.asyncio
async def test_data_ids_are_refused_while_connected_to_a_remote_instance():
    remote = SimpleNamespace(cognify=AsyncMock())
    with (
        patch.object(serve_state, "get_remote_client", return_value=remote),
        pytest.raises(ValueError, match="remote Cognee instance"),
    ):
        await cognify_module.cognify(datasets=["repos"], data_ids=[REPO_A.id])

    remote.cognify.assert_not_awaited()
