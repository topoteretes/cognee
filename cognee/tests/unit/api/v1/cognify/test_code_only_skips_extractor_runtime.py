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


async def _cognify(
    items,
    *,
    incremental_loading=True,
    runtime=None,
    execute_run=None,
    extractor="gliner_demo",
    read_rows=None,
    **cognify_kwargs,
):
    """Run cognify keyless (gliner_demo resolved, runtime missing) over ``items``."""
    if runtime is None:
        runtime = AsyncMock(side_effect=KeylessExtractorNotInstalledError())

    async def execute(**kwargs):
        if "before_run" in kwargs:
            await kwargs["before_run"](items, DATASET)
        return {}

    pipeline = AsyncMock(side_effect=execute_run or execute)
    environment = AsyncMock()
    pipeline.environment = environment

    with (
        patch.object(cognify_module, "resolve_extractor", return_value=extractor),
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
            read_rows or AsyncMock(return_value=items),
        ),
        patch.object(cognify_module, "get_pipeline_executor", return_value=pipeline),
        patch(
            "cognee.modules.pipelines.layers.setup_and_check_environment.setup_and_check_environment",
            environment,
        ),
    ):
        try:
            await cognify_module.cognify(
                datasets=["my_code"],
                incremental_loading=incremental_loading,
                raise_on_error=False,
                **cognify_kwargs,
            )
        except KeylessExtractorNotInstalledError:
            pass
    return runtime, pipeline


@pytest.mark.asyncio
async def test_code_only_dataset_never_ensures_the_extractor_runtime():
    runtime, pipeline = await _cognify([CODE_FILE, CODE_REPO])

    runtime.assert_not_awaited()
    pipeline.assert_awaited_once()
    kwargs = pipeline.environment.await_args.kwargs
    assert kwargs["needs_llm"] is False
    assert kwargs["skip_connection_test"] is True


@pytest.mark.parametrize(
    "items, incremental_loading, needs_runtime",
    [
        # Incremental loading skips the completed document, leaving only code.
        ([CODE_REPO, _item(completed=True)], True, False),
        # Without incremental loading the completed document is rebuilt.
        ([CODE_REPO, _item(completed=True)], False, True),
        # An unchanged, already-built repository re-remembered: nothing pending.
        ([_item(source="code_repo", completed=True)], True, False),
    ],
    ids=["completed_document_skipped", "no_incremental_loading", "nothing_pending"],
)
@pytest.mark.asyncio
async def test_completed_items_count_only_without_incremental_loading(
    items, incremental_loading, needs_runtime
):
    runtime, pipeline = await _cognify(items, incremental_loading=incremental_loading)

    assert runtime.await_count == int(needs_runtime)
    assert pipeline.environment.await_count == int(not needs_runtime)


@pytest.mark.asyncio
async def test_mixed_dataset_still_ensures_the_extractor_runtime():
    runtime = AsyncMock()
    _, pipeline = await _cognify([CODE_FILE, _item()], runtime=runtime)

    assert runtime.await_count == 2
    assert runtime.await_args.args[1].gliner_auto_install is False
    assert pipeline.environment.await_args.kwargs["skip_connection_test"] is False


@pytest.mark.asyncio
async def test_runtime_installation_precedes_the_dataset_lock():
    from cognee.infrastructure.locks import dataset_lock
    from cognee.infrastructure.locks.dataset_lock import held_datasets

    calls = []

    async def ensure(extractor, config):
        locked = DATASET.id in held_datasets.get()
        calls.append((locked, config.gliner_auto_install))
        assert not locked or config.gliner_auto_install is False

    async def execute(**kwargs):
        async with dataset_lock(DATASET.id):
            await kwargs["before_run"]([_item()], DATASET)
        return {}

    runtime = AsyncMock(side_effect=ensure)
    # _cognify's executor is overridden here to take the real lock.
    await _cognify([_item()], runtime=runtime, execute_run=execute)
    assert calls == [(False, True), (True, False)]


@pytest.mark.asyncio
async def test_the_llm_extractor_never_reads_the_rows_to_look_for_a_runtime():
    """Only GLiNER has a runtime to install, so an LLM cognify skips the whole-dataset scan."""
    runtime = AsyncMock()
    read_rows = AsyncMock(return_value=[_item()])

    await _cognify([_item()], extractor="llm", runtime=runtime, read_rows=read_rows)

    # The locked run's own check still asks, and is a no-op for the LLM extractor.
    assert all(call.args[0] == "llm" for call in runtime.await_args_list)
    assert read_rows.await_count == 0  # the preflight scan of the dataset's rows


def test_code_only_classification_uses_selected_rows_only():
    assert cognify_module._pending_items_are_code_only([CODE_REPO], DATASET.id, True)
    assert not cognify_module._pending_items_are_code_only([CODE_REPO, _item()], DATASET.id, True)


@pytest.mark.asyncio
async def test_the_gliner_preflight_reads_only_the_scoped_rows():
    row = _item()
    read_rows = AsyncMock(return_value=[row])
    runtime = AsyncMock()

    await _cognify([row], runtime=runtime, read_rows=read_rows, data_ids=[row.id])

    # The preflight looked at the selected row (a document, so it asked for the
    # runtime) and never read the dataset whole.
    assert runtime.await_count == 2  # the preflight, then the run under the lock
    assert all(call.kwargs.get("data_ids") for call in read_rows.await_args_list)
