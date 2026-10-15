"""``cognify(data_ids=...)`` builds only those items, in a run of their own.

remember() builds each declared repository this way, so a repository whose
build fails rolls back its own ``cognify_pipeline`` run only, and an earlier
repository still pending in the dataset is not rebuilt by every later one.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

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


async def _cognify(datasets=("repos",), **cognify_kwargs):
    pipeline = AsyncMock(return_value={DATASET.id: SimpleNamespace(status="done")})
    runtime = AsyncMock()

    with (
        patch.object(cognify_module, "ensure_extractor_runtime", runtime),
        patch.object(migrations, "run_migrations_and_block", AsyncMock()),
        patch(
            "cognee.modules.users.methods.get_default_user",
            AsyncMock(return_value=SimpleNamespace(id=uuid4())),
        ),
        patch(
            "cognee.modules.data.methods.get_authorized_existing_datasets",
            AsyncMock(return_value=[DATASET]),
        ),
        patch(
            "cognee.modules.data.methods.get_dataset_data",
            AsyncMock(return_value=[REPO_A, REPO_B, DOCUMENT]),
        ),
        patch.object(cognify_module, "get_pipeline_executor", return_value=pipeline),
    ):
        result = await cognify_module.cognify(datasets=list(datasets), **cognify_kwargs)
    return result, pipeline, runtime


@pytest.mark.asyncio
async def test_data_ids_scope_the_run_to_those_rows():
    _result, pipeline, runtime = await _cognify(data_ids=[REPO_B.id])

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


@pytest.mark.asyncio
async def test_ids_outside_the_dataset_run_nothing():
    # An empty data list would make the runner fall back to the whole dataset.
    result, pipeline, _runtime = await _cognify(data_ids=[uuid4()])

    assert result == {}
    pipeline.assert_not_awaited()


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
    with pytest.raises(ValueError, match=message):
        await _cognify(data_ids=[REPO_A.id], **cognify_kwargs)
