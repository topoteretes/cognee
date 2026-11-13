"""When ``cognify()`` lets another process rebuild its task lists, and with what.

A data item executor (``cognee.modules.pipelines.execution``) may run a run's
items in other processes, which share this one's environment but none of its
memory or local files. ``cognify()`` marks a resolver portable only for a
call another process would rebuild identically, and sends the settings it
resolved rather than the arguments as given. That holds for both of its runs:
the cognify run and the indexing run before it.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cognee.api.v1.cognify.cognify
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.routing import CognifyRoute
from cognee.modules.pipelines.execution import build_task_resolver
from cognee.shared.data_models import KnowledgeGraph

cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]


class _OtherChunker(TextChunker):
    pass


class _OtherGraph(KnowledgeGraph):
    pass


async def _resolver_of(**cognify_kwargs):
    """The resolver ``cognify()`` hands the pipeline, and how the task lists were built."""
    call, default_tasks_kwargs = await _pipeline_call_of(**cognify_kwargs)
    return call["tasks"], default_tasks_kwargs


async def _indexing_resolver_of(**cognify_kwargs):
    """The resolver ``cognify()`` hands the pipeline for its indexing run."""
    call, _ = await _pipeline_call_of(**cognify_kwargs)
    return call["indexing_tasks"]


async def _pipeline_call_of(**cognify_kwargs):
    """What ``cognify()`` calls the pipeline with, and how the task lists were built."""
    calls = []

    async def fake_executor(**kwargs):
        calls.append(kwargs)
        return {"ds": "run_info"}

    default_tasks = AsyncMock(return_value="STANDARD_TASKS")
    with (
        patch.object(
            cognify_module, "get_pipeline_executor", lambda run_in_background: fake_executor
        ),
        patch.object(cognify_module, "get_default_tasks", new=default_tasks),
        patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value="DLT_TASKS")),
        patch.object(cognify_module, "get_code_file_tasks", new=MagicMock(return_value="CODE")),
        patch.object(cognify_module, "get_code_repo_tasks", new=MagicMock(return_value="REPO")),
        patch.object(cognify_module, "get_indexing_tasks", new=AsyncMock(return_value="INDEX")),
        patch.object(
            cognify_module, "get_configured_ontology_resolver", new=MagicMock(return_value=None)
        ),
    ):
        await cognify_module.cognify(datasets=["ds"], **cognify_kwargs)
    (call,) = calls
    return call, default_tasks.await_args.kwargs


@pytest.mark.asyncio
async def test_plain_call_is_portable_with_the_settings_it_resolved():
    resolver, _ = await _resolver_of(chunk_size=1024, early_chunk_indexing=False)

    name, args = resolver.portable
    assert name == cognify_module.PORTABLE_COGNIFY_RESOLVER
    assert args["chunk_size"] == 1024
    # Resolved here, not None: a worker must not resolve them from its own state.
    assert isinstance(args["chunks_per_batch"], int)
    assert args["early_chunk_indexing"] is False
    assert args["summary_method"] in ("llm", "from_extraction")
    assert isinstance(args["extractor"], str)
    assert args["custom_prompt"] is None
    # Nothing in it names a file or an object of this process.
    assert set(args) == {
        "chunk_size",
        "chunks_per_batch",
        "custom_prompt",
        "summary_method",
        "early_chunk_indexing",
        "extractor",
    }


@pytest.mark.asyncio
async def test_unset_settings_travel_resolved(monkeypatch):
    async def resolved_chunk_size(chunk_size):
        return 777 if chunk_size is None else chunk_size

    monkeypatch.setattr(cognify_module, "resolve_chunk_size", resolved_chunk_size)
    monkeypatch.setattr(cognify_module, "resolve_chunks_per_batch", lambda value: value or 33)

    resolver, _ = await _resolver_of(early_chunk_indexing=True)

    args = resolver.portable[1]
    assert (args["chunk_size"], args["chunks_per_batch"]) == (777, 33)
    assert args["early_chunk_indexing"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"graph_model": _OtherGraph},
        {"chunker": _OtherChunker},
        {"config": {"ontology_config": {"ontology_resolver": None}}},
        {"ontology_file_path": "/tmp/on-this-machine.owl"},
        {"vector_db_config": {"vector_db_provider": "pgvector"}},
        {"graph_db_config": {"graph_database_provider": "postgres"}},
        {"some_extraction_kwarg": 1},
    ],
    ids=[
        "graph_model",
        "chunker",
        "ontology config",
        "ontology file",
        "vector config",
        "graph config",
        "extraction kwargs",
    ],
)
async def test_call_another_process_could_not_rebuild_is_not_portable(kwargs):
    resolver, _ = await _resolver_of(**kwargs)

    assert resolver.portable is None


def _described(tasks):
    """What a task list does, comparable between two processes' builds of it."""

    def plain(value):
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in sorted(value.items(), key=str)}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return getattr(value, "__name__", type(value).__name__)

    return [
        (
            getattr(task.executable, "__name__", str(task.executable)),
            plain(task.task_config),
            plain(task.default_params.get("kwargs")),
            task.needs_llm,
        )
        for task in tasks
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cognify_kwargs",
    [
        {},
        {"chunk_size": 512},
        {"chunks_per_batch": 7},
        {"early_chunk_indexing": False},
        {"early_chunk_indexing": True},
        {"summary_method": "llm", "custom_prompt": "be brief"},
        {"extractor": "llm"},
    ],
    ids=str,
)
async def test_another_process_rebuilds_the_standard_task_list_the_owner_built(cognify_kwargs):
    """With the real task builders on both sides: the owner builds from the
    arguments as given, the worker from what the owner resolved, and the two
    lists must do the same thing. Real builders, because with stand-ins this
    compares two constants and passes while the real lists differ."""
    calls = []

    async def fake_executor(**kwargs):
        calls.append(kwargs)
        return {"ds": "run_info"}

    async def chunk_size_of(chunk_size):
        return chunk_size or 1024

    with (
        patch.object(
            cognify_module, "get_pipeline_executor", lambda run_in_background: fake_executor
        ),
        patch.object(cognify_module, "resolve_chunk_size", chunk_size_of),
        patch("cognee.infrastructure.llm.resolve_chunk_size", chunk_size_of),
        patch.object(cognify_module, "get_indexing_tasks", new=AsyncMock(return_value=[])),
        patch.object(
            cognify_module, "get_configured_ontology_resolver", new=MagicMock(return_value=None)
        ),
        patch.object(cognify_module, "ensure_extractor_runtime", new=AsyncMock()),
    ):
        await cognify_module.cognify(datasets=["ds"], **cognify_kwargs)
        owner = calls[0]["tasks"]
        _, args = owner.portable
        worker = await cognify_module.build_portable_cognify_task_resolver(user=object(), **args)

    assert _described(worker.tasks_by_route[CognifyRoute.STANDARD]) == _described(
        owner.tasks_by_route[CognifyRoute.STANDARD]
    )


@pytest.mark.asyncio
async def test_another_process_knows_the_standard_route_only():
    """A code repository is a directory on the owner's disk, and the DLT and code
    lists read defaults the portable arguments do not carry, so those routes are
    not rebuilt: an item of theirs sent to another process fails there by name
    instead of running a list that differs from the owner's."""
    resolver, _ = await _resolver_of()
    _, args = resolver.portable

    with (
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value="STANDARD")),
        patch.object(cognify_module, "ensure_extractor_runtime", new=AsyncMock()),
        patch.object(
            cognify_module, "get_configured_ontology_resolver", new=MagicMock(return_value=None)
        ),
    ):
        rebuilt = await cognify_module.build_portable_cognify_task_resolver(user=object(), **args)

    assert rebuilt.tasks_by_route == {CognifyRoute.STANDARD: "STANDARD"}
    assert set(resolver.tasks_by_route) == set(CognifyRoute)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", list(CognifyRoute))
async def test_only_standard_items_are_offered_to_another_process(route):
    """The owner's resolver says which items the rebuilt one can run, so an
    executor never sends an item to a process that would fail it by name."""
    resolver, _ = await _resolver_of()

    with patch.object(cognify_module, "cognify_route_for", return_value=route):
        assert resolver.portable_item(object()) is (route is CognifyRoute.STANDARD)


@pytest.mark.asyncio
async def test_indexing_run_is_portable_with_the_chunking_the_call_resolved(monkeypatch):
    async def resolved_chunk_size(chunk_size):
        return 777 if chunk_size is None else chunk_size

    monkeypatch.setattr(cognify_module, "resolve_chunk_size", resolved_chunk_size)
    monkeypatch.setattr(cognify_module, "resolve_chunks_per_batch", lambda value: value or 33)

    resolver = await _indexing_resolver_of(early_chunk_indexing=True)

    # All the indexing run depends on, and what the cognify run is sent too: both
    # processes' runs must cut the same chunks.
    assert resolver.portable == (
        cognify_module.PORTABLE_INDEXING_RESOLVER,
        {"chunk_size": 777, "chunks_per_batch": 33},
    )
    cognify_resolver, _ = await _resolver_of(early_chunk_indexing=True)
    assert resolver.portable[1].items() <= cognify_resolver.portable[1].items()


@pytest.mark.asyncio
async def test_call_without_early_chunk_indexing_has_no_indexing_run():
    assert await _indexing_resolver_of(early_chunk_indexing=False) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "portable"),
    [
        # What the indexing run does not read does not keep it in this process.
        ({"graph_model": _OtherGraph}, True),
        ({"config": {"ontology_config": {"ontology_resolver": None}}}, True),
        ({"ontology_file_path": "/tmp/on-this-machine.owl"}, True),
        ({"some_extraction_kwarg": 1}, True),
        ({"custom_prompt": "be brief"}, True),
        # What it cuts its chunks with, and where it stores them, does.
        ({"chunker": _OtherChunker}, False),
        ({"vector_db_config": {"vector_db_provider": "pgvector"}}, False),
        ({"graph_db_config": {"graph_database_provider": "postgres"}}, False),
    ],
    ids=[
        "graph_model",
        "ontology config",
        "ontology file",
        "extraction kwargs",
        "custom prompt",
        "chunker",
        "vector config",
        "graph config",
    ],
)
async def test_indexing_run_is_portable_when_another_process_cuts_the_same_chunks(kwargs, portable):
    resolver = await _indexing_resolver_of(early_chunk_indexing=True, **kwargs)

    assert (resolver.portable is not None) is portable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cognify_kwargs",
    [{}, {"chunk_size": 512}, {"chunks_per_batch": 7}],
    ids=str,
)
async def test_another_process_rebuilds_the_indexing_task_list_the_owner_built(cognify_kwargs):
    """With the real task builder on both sides, and the worker's reached by the
    name the owner sends (so the factory is registered under it)."""
    calls = []

    async def fake_executor(**kwargs):
        calls.append(kwargs)
        return {"ds": "run_info"}

    async def chunk_size_of(chunk_size):
        return chunk_size or 1024

    with (
        patch.object(
            cognify_module, "get_pipeline_executor", lambda run_in_background: fake_executor
        ),
        patch.object(cognify_module, "resolve_chunk_size", chunk_size_of),
        patch("cognee.infrastructure.llm.resolve_chunk_size", chunk_size_of),
        patch.object(
            cognify_module, "get_configured_ontology_resolver", new=MagicMock(return_value=None)
        ),
        patch.object(cognify_module, "ensure_extractor_runtime", new=AsyncMock()),
    ):
        await cognify_module.cognify(datasets=["ds"], early_chunk_indexing=True, **cognify_kwargs)
        owner = calls[0]["indexing_tasks"]
        name, args = owner.portable
        worker = await build_task_resolver(name, args, user=object())

    owner_tasks = owner.tasks_by_route[CognifyRoute.STANDARD]
    assert owner_tasks and _described(worker.tasks_by_route[CognifyRoute.STANDARD]) == _described(
        owner_tasks
    )
    # The rebuilt resolver knows the standard route only, the one route the
    # owner's indexing run has work for.
    assert set(worker.tasks_by_route) == {CognifyRoute.STANDARD}
    assert [route for route, tasks in owner.tasks_by_route.items() if tasks] == [
        CognifyRoute.STANDARD
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", list(CognifyRoute))
async def test_only_standard_items_of_the_indexing_run_are_offered_to_another_process(route):
    resolver = await _indexing_resolver_of(early_chunk_indexing=True)

    with patch.object(cognify_module, "cognify_route_for", return_value=route):
        assert resolver.portable_item(object()) is (route is CognifyRoute.STANDARD)
