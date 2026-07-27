"""Unit tests for when ``memify`` projects the memory fragment.

Everything is mocked: no graph backend, no vector backend, no network. These
tests pin *whether* ``get_memory_fragment`` is called, because projecting it
reads the entire graph into memory and the default configuration had no
consumer for the result.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.memify.memify import memify


def _make_async_ctx_mock():
    """Build a ``MagicMock`` that behaves as an async-context-manager factory."""
    inner = MagicMock()
    inner.__aenter__ = AsyncMock(return_value=inner)
    inner.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=inner)


def _patches(*, extraction_tasks, fragment=None):
    """Patch every backend boundary ``memify`` touches.

    Returns (context_managers, handles) so each test can assert on the mocks it
    cares about.
    """
    authorized_dataset = SimpleNamespace(id="dataset-1", owner_id="owner-1")

    get_fragment = AsyncMock(return_value=fragment or MagicMock(name="CogneeGraph"))
    executor = AsyncMock(return_value={"status": "ok"})

    handles = {"get_fragment": get_fragment, "executor": executor}

    ctxs = [
        patch("cognee.modules.memify.memify.setup", new=AsyncMock()),
        patch(
            "cognee.modules.memify.memify.resolve_authorized_user_datasets",
            new=AsyncMock(return_value=(MagicMock(), [authorized_dataset])),
        ),
        patch(
            "cognee.modules.memify.memify.set_database_global_context_variables",
            new=_make_async_ctx_mock(),
        ),
        patch("cognee.modules.memify.memify.get_memory_fragment", new=get_fragment),
        patch(
            "cognee.modules.memify.memify.get_default_memify_extraction_tasks",
            return_value=extraction_tasks,
        ),
        patch(
            "cognee.modules.memify.memify.get_default_memify_enrichment_tasks",
            return_value=[MagicMock(name="index_data_points_task")],
        ),
        patch(
            "cognee.modules.memify.memify.get_pipeline_executor",
            return_value=executor,
        ),
    ]
    return ctxs, handles


@pytest.mark.asyncio
async def test_skips_fragment_projection_when_no_extraction_tasks():
    """The default config (triplet_embedding off) resolves to no extraction tasks.

    The fragment would go straight to the default enrichment task, which skips
    non-DataPoint objects -- so building it is a wasted full-graph read.
    """
    ctxs, handles = _patches(extraction_tasks=[])

    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4], ctxs[5], ctxs[6]:
        result = await memify()

    assert result == {"status": "ok"}
    handles["get_fragment"].assert_not_awaited()
    # The pipeline still runs; it simply receives no fragment.
    assert handles["executor"].await_args.kwargs["data"] is None


@pytest.mark.asyncio
async def test_projects_fragment_when_extraction_tasks_exist():
    """With triplet_embedding on there is a real consumer, so keep projecting."""
    fragment = MagicMock(name="CogneeGraph")
    ctxs, handles = _patches(
        extraction_tasks=[MagicMock(name="get_triplet_datapoints_task")],
        fragment=fragment,
    )

    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4], ctxs[5], ctxs[6]:
        await memify()

    handles["get_fragment"].assert_awaited_once()
    assert handles["executor"].await_args.kwargs["data"] == [fragment]


@pytest.mark.asyncio
async def test_projects_fragment_for_caller_supplied_enrichment_only_pipeline():
    """No regression for an enrichment-only pipeline the caller built itself.

    Such a pipeline may consume the fragment directly, so it must still get one
    even though no extraction tasks resolve.
    """
    fragment = MagicMock(name="CogneeGraph")
    ctxs, handles = _patches(extraction_tasks=[], fragment=fragment)

    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4], ctxs[5], ctxs[6]:
        await memify(enrichment_tasks=[MagicMock(name="custom_enrichment_task")])

    handles["get_fragment"].assert_awaited_once()
    assert handles["executor"].await_args.kwargs["data"] == [fragment]


@pytest.mark.asyncio
async def test_explicit_data_is_never_overwritten_by_a_projection():
    """Caller-supplied data short-circuits the projection, as before."""
    ctxs, handles = _patches(extraction_tasks=[MagicMock(name="task")])

    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4], ctxs[5], ctxs[6]:
        await memify(data=["explicit"])

    handles["get_fragment"].assert_not_awaited()
    assert handles["executor"].await_args.kwargs["data"] == ["explicit"]
