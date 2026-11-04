"""A keyless cognify over code-only data never needs the GLiNER runtime (SDK-793).

Code files and code-repo manifests take the CODE / CODE_REPO routes, which run enola
only. A keyless install without gliner2 must still build their graphs, so cognify()
must neither resolve the graph extractor, install its runtime, nor raise
KeylessExtractorNotInstalledError for them, and it probes neither the LLM nor the
embedding provider unless it indexes vectors. Any non-code item still pending keeps
every check. ``needs_llm`` comes from the routes the pending items take.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from cognee.modules.cognify.config import KeylessExtractorNotInstalledError
from cognee.modules.cognify.routing import CognifyRoute
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
    items, *, incremental_loading=True, runtime=None, extractor="gliner_demo", **cognify_kwargs
):
    """Run cognify keyless (gliner_demo resolved, runtime missing) over ``items``."""
    if runtime is None:
        runtime = AsyncMock(side_effect=KeylessExtractorNotInstalledError())
    pipeline = AsyncMock(return_value={})
    resolve = Mock(return_value=extractor)

    with (
        patch.object(cognify_module, "resolve_extractor", resolve),
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
    return SimpleNamespace(runtime=runtime, pipeline=pipeline, resolve=resolve)


@pytest.mark.asyncio
async def test_code_only_dataset_never_resolves_or_installs_the_extractor():
    run = await _cognify([CODE_FILE, CODE_REPO])

    run.resolve.assert_not_called()
    run.runtime.assert_not_awaited()
    run.pipeline.assert_awaited_once()
    kwargs = run.pipeline.await_args.kwargs
    assert kwargs["needs_llm"] is False
    assert kwargs["skip_connection_test"] is True
    assert kwargs["extras"] == {"graph_extractor": None}


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
    run = await _cognify(items, incremental_loading=incremental_loading)

    assert run.resolve.call_count == int(needs_runtime)
    assert run.runtime.await_count == int(needs_runtime)
    assert run.pipeline.await_count == int(not needs_runtime)


@pytest.mark.asyncio
async def test_mixed_dataset_still_resolves_the_extractor_and_probes():
    runtime = AsyncMock()
    run = await _cognify([CODE_FILE, _item()], runtime=runtime)

    run.resolve.assert_called_once()
    runtime.assert_awaited_once()
    kwargs = run.pipeline.await_args.kwargs
    assert kwargs["skip_connection_test"] is False
    assert kwargs["extras"] == {"graph_extractor": "gliner_demo"}


@pytest.mark.asyncio
async def test_code_only_run_with_index_vectors_still_probes_embeddings():
    run = await _cognify([CODE_REPO], codegraph_config={"index_vectors": True})

    kwargs = run.pipeline.await_args.kwargs
    assert kwargs["needs_llm"] is False
    assert kwargs["skip_connection_test"] is False


@pytest.mark.asyncio
async def test_needs_llm_follows_the_pending_routes_not_every_wired_route():
    """A pending document on the LLM extractor needs the probe; with an unreadable
    pending set the union over every wired route decides, as before."""
    runtime = AsyncMock()
    with patch.object(cognify_module, "_pending_routes", AsyncMock(return_value=None)):
        run = await _cognify([CODE_REPO], runtime=runtime)
    assert run.pipeline.await_args.kwargs["needs_llm"] is False  # gliner: no LLM task
    assert run.pipeline.await_args.kwargs["skip_connection_test"] is False

    with patch.object(cognify_module, "_pending_routes", AsyncMock(return_value=None)):
        run = await _cognify([CODE_REPO], runtime=runtime, extractor="llm")
    assert run.pipeline.await_args.kwargs["needs_llm"] is True


@pytest.mark.asyncio
async def test_pending_routes_are_read_from_the_datasets_rows():
    other = SimpleNamespace(id=uuid4(), name="other")
    rows = AsyncMock(side_effect=[[_item(), CODE_FILE], [CODE_REPO]])

    with (
        patch(
            "cognee.modules.data.methods.get_authorized_existing_datasets",
            AsyncMock(return_value=[DATASET, other]),
        ),
        patch("cognee.modules.data.methods.get_dataset_data", rows),
    ):
        routes = await cognify_module._pending_routes(
            ["my_code", "other"], SimpleNamespace(id=uuid4()), True
        )

    assert routes == {CognifyRoute.STANDARD, CognifyRoute.CODE, CognifyRoute.CODE_REPO}
    assert rows.await_count == 2


@pytest.mark.asyncio
async def test_an_unreadable_pending_set_answers_none():
    with patch(
        "cognee.modules.data.methods.get_authorized_existing_datasets",
        AsyncMock(side_effect=PermissionError("no")),
    ):
        assert (
            await cognify_module._pending_routes(["x"], SimpleNamespace(id=uuid4()), True) is None
        )
