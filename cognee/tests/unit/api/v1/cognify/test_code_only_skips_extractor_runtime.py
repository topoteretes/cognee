"""A keyless cognify over code-only data never needs the GLiNER runtime (SDK-795).

Code files and code-repo manifests take the CODE / CODE_REPO routes, which run enola
only. A keyless install without gliner2 must still build their graphs, so cognify()
must neither install the runtime nor raise KeylessExtractorNotInstalledError for
them. Any non-code item still pending keeps the runtime check.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.cognify.config import KeylessExtractorNotInstalledError
from cognee.modules.pipelines.models import DataItemStatus

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
migrations = importlib.import_module("cognee.modules.migrations.startup")

DATASET = SimpleNamespace(id=uuid4(), name="my_code")


def _item(source=None, extension="txt", completed=False):
    status = {}
    if completed:
        status = {
            "cognify_pipeline": {
                str(DATASET.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED.value
            }
        }
    return SimpleNamespace(
        id=uuid4(),
        extension=extension,
        system_metadata={"source": source} if source else None,
        pipeline_status=status,
    )


CODE_FILE = _item(source="code", extension="py")
CODE_REPO = _item(source="code_repo", extension="json")


async def _cognify(items, *, incremental_loading=True, runtime=None, **cognify_kwargs):
    """Run cognify keyless (gliner_demo resolved, runtime missing) over ``items``."""
    if runtime is None:
        runtime = AsyncMock(side_effect=KeylessExtractorNotInstalledError())
    pipeline = AsyncMock(return_value={})

    with (
        patch.object(cognify_module, "resolve_extractor", return_value="gliner_demo"),
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
            AsyncMock(return_value=items),
        ),
        patch.object(cognify_module, "get_pipeline_executor", return_value=pipeline),
    ):
        try:
            await cognify_module.cognify(
                datasets=["my_code"], incremental_loading=incremental_loading, **cognify_kwargs
            )
        except KeylessExtractorNotInstalledError:
            pass
    return runtime, pipeline


@pytest.mark.asyncio
async def test_code_only_dataset_never_ensures_the_extractor_runtime():
    runtime, pipeline = await _cognify([CODE_FILE, CODE_REPO])

    runtime.assert_not_awaited()
    pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_completed_documents_do_not_require_the_runtime_under_incremental_loading():
    runtime, pipeline = await _cognify([CODE_REPO, _item(completed=True)])

    runtime.assert_not_awaited()
    pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_completed_documents_still_count_without_incremental_loading():
    runtime, pipeline = await _cognify(
        [CODE_REPO, _item(completed=True)], incremental_loading=False
    )

    runtime.assert_awaited_once()
    pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_dataset_still_ensures_the_extractor_runtime():
    runtime, pipeline = await _cognify([CODE_FILE, _item()])

    runtime.assert_awaited_once()
    pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_nothing_pending_needs_no_runtime():
    # An unchanged, already-built repository re-remembered: incremental loading
    # skips it, so the extractor never runs.
    runtime, pipeline = await _cognify([_item(source="code_repo", completed=True)])

    runtime.assert_not_awaited()
    pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failing_lookup_keeps_the_runtime_check():
    runtime = AsyncMock()
    with (
        patch.object(cognify_module, "resolve_extractor", return_value="gliner_demo"),
        patch.object(cognify_module, "ensure_extractor_runtime", runtime),
        patch.object(migrations, "run_migrations_and_block", AsyncMock()),
        patch(
            "cognee.modules.users.methods.get_default_user",
            AsyncMock(side_effect=RuntimeError("db down")),
        ),
        patch.object(
            cognify_module, "get_pipeline_executor", return_value=AsyncMock(return_value={})
        ),
    ):
        await cognify_module.cognify(datasets=["my_code"])

    runtime.assert_awaited_once()


@pytest.mark.asyncio
async def test_code_only_run_skips_the_provider_probes():
    _, pipeline = await _cognify([CODE_FILE, CODE_REPO])

    kwargs = pipeline.await_args.kwargs
    assert kwargs["needs_llm"] is False
    assert kwargs["skip_connection_test"] is True


@pytest.mark.asyncio
async def test_code_only_run_with_index_vectors_still_probes_embeddings():
    _, pipeline = await _cognify([CODE_REPO], codegraph_config={"index_vectors": True})

    kwargs = pipeline.await_args.kwargs
    assert kwargs["needs_llm"] is False
    assert kwargs["skip_connection_test"] is False


@pytest.mark.asyncio
async def test_mixed_run_keeps_the_provider_probes():
    runtime = AsyncMock()
    _, pipeline = await _cognify([CODE_FILE, _item()], runtime=runtime)

    runtime.assert_awaited_once()
    assert pipeline.await_args.kwargs["skip_connection_test"] is False
