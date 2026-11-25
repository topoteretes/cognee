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
from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.routing import CognifyRoute
from cognee.modules.pipelines.execution import build_task_resolver
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.graph_model_utils import graph_model_to_graph_schema, graph_schema_to_graph_model

cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]


class _OtherChunker(TextChunker):
    pass


class _OtherGraph(KnowledgeGraph):
    """A custom graph model that is no ``DataPoint``: another process cannot
    build it from its schema."""


class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class PeopleGraph(DataPoint):
    """A custom graph model written in Python: a run given it stays in its process."""

    people: list[Person] = []
    metadata: dict = {"index_fields": []}


def _people_graph_from_its_schema() -> type[DataPoint]:
    """The custom graph model as it arrives over the API: built from a JSON
    schema. Another process builds the same classes from its schema again."""
    return graph_schema_to_graph_model(graph_model_to_graph_schema(PeopleGraph))


FROM_ITS_SCHEMA = "a graph model built from its schema"


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
@pytest.mark.parametrize("graph_model", [None, KnowledgeGraph], ids=["None", "KnowledgeGraph"])
async def test_the_default_graph_model_is_portable_however_it_is_given(graph_model):
    """``None`` is the default graph model too: what decides is the model the
    call resolved, not the value it was given."""
    resolver, default_tasks_kwargs = await _resolver_of(graph_model=graph_model)

    assert resolver.portable is not None
    assert default_tasks_kwargs["graph_model"] is KnowledgeGraph


@pytest.mark.asyncio
async def test_plain_call_is_portable_with_the_settings_it_resolved():
    resolver, _ = await _resolver_of(chunk_size=1024, early_chunk_indexing=False)

    name, args = resolver.portable
    assert name == cognify_module.PORTABLE_COGNIFY_RESOLVER
    assert resolver.stays_because is None
    assert args["chunk_size"] == 1024
    # Resolved here, not None: a worker must not resolve them from its own state.
    assert isinstance(args["chunks_per_batch"], int)
    assert args["early_chunk_indexing"] is False
    assert args["summary_method"] in ("llm", "from_extraction")
    assert isinstance(args["extractor"], str)
    assert args["custom_prompt"] is None
    # The default KnowledgeGraph travels as no schema at all.
    assert args["graph_model_schema"] is None
    assert args["chunk_attachment"] is None
    # Nothing in it names a file or an object of this process.
    assert set(args) == {
        "chunk_size",
        "chunks_per_batch",
        "custom_prompt",
        "summary_method",
        "early_chunk_indexing",
        "extractor",
        "graph_model_schema",
        "chunk_attachment",
        "dlt_chunks_per_batch",
    }
    # The DLT list stores 100 rows at a time unless told otherwise, which is
    # not the standard list's batch: it travels resolved, by itself.
    assert args["dlt_chunks_per_batch"] == 100


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


GRAPH_MODEL_STAYS = (
    "another process cannot build its graph model (a class written in Python, "
    "or a schema that names a class it does not define)"
)
STORE_CONFIGURATION = "the call passes a vector or graph database configuration"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, stays_because",
    [
        ({"graph_model": _OtherGraph}, GRAPH_MODEL_STAYS),
        ({"chunker": _OtherChunker}, "the call passes a chunker of its own"),
        (
            {"config": {"ontology_config": {"ontology_resolver": None}}},
            "the call passes an ontology configuration of its own",
        ),
        ({"ontology_file_path": "/tmp/on-this-machine.owl"}, "the call passes an ontology file"),
        ({"vector_db_config": {"vector_db_provider": "pgvector"}}, STORE_CONFIGURATION),
        ({"graph_db_config": {"graph_database_provider": "postgres"}}, STORE_CONFIGURATION),
        (
            {"some_extraction_kwarg": 1, "another_one": 2},
            "the call passes extraction arguments of its own (another_one, some_extraction_kwarg)",
        ),
    ],
    ids=[
        "graph_model that is no DataPoint",
        "chunker",
        "ontology config",
        "ontology file",
        "vector config",
        "graph config",
        "extraction kwargs",
    ],
)
async def test_call_another_process_could_not_rebuild_is_not_portable_and_says_why(
    kwargs, stays_because
):
    """The reason is what a run that stays in its process is logged with."""
    resolver, _ = await _resolver_of(**kwargs)

    assert resolver.portable is None
    assert resolver.stays_because == stays_because


@pytest.mark.asyncio
async def test_a_graph_model_built_from_a_schema_travels_as_its_schema():
    """Another process builds a graph model that was built from a JSON schema
    from its schema again, so such a call stays portable, and with it the chunk
    attachment only a custom model takes."""
    graph_model = _people_graph_from_its_schema()
    resolver, _ = await _resolver_of(graph_model=graph_model, chunk_attachment="all")

    name, args = resolver.portable
    assert name == cognify_module.PORTABLE_COGNIFY_RESOLVER
    assert args["graph_model_schema"] == graph_model_to_graph_schema(graph_model)
    assert args["chunk_attachment"] == "all"


@pytest.mark.asyncio
async def test_a_graph_model_given_as_a_schema_travels_as_a_schema():
    """``cognify()`` builds a schema it is given into a class, and another
    process builds that class from its schema again: what decides is the model
    the call resolved, not the dict it was given."""
    resolver, default_tasks_kwargs = await _resolver_of(
        graph_model=graph_model_to_graph_schema(PeopleGraph)
    )

    _, args = resolver.portable
    assert resolver.stays_because is None
    assert args["graph_model_schema"] == graph_model_to_graph_schema(
        default_tasks_kwargs["graph_model"]
    )


@pytest.mark.asyncio
async def test_a_graph_model_written_in_python_keeps_the_run_in_its_process():
    """Its schema does not build the class it was (a field with a default comes
    back optional), so another process would extract with another model."""
    resolver, _ = await _resolver_of(graph_model=PeopleGraph, chunk_attachment="all")

    assert resolver.portable is None
    assert resolver.stays_because == GRAPH_MODEL_STAYS


def test_a_graph_model_built_from_a_schema_builds_the_same_schema_again():
    """What the rule above rests on: once built from a schema, a model's schema
    is the one a model built from it has."""
    graph_model = _people_graph_from_its_schema()
    schema = graph_model_to_graph_schema(graph_model)

    assert graph_model_to_graph_schema(graph_schema_to_graph_model(schema)) == schema
    # And not before: this is why a class written in Python stays.
    assert graph_model_to_graph_schema(PeopleGraph) != schema


def test_which_graph_models_another_process_can_build():
    schema_for = cognify_module.graph_model_schema_for_workers

    # The default needs no schema: the worker has it.
    assert schema_for(KnowledgeGraph) == (True, None)
    # A model built from a schema goes as its schema.
    graph_model = _people_graph_from_its_schema()
    assert schema_for(graph_model) == (True, graph_model_to_graph_schema(graph_model))
    # A class written in Python does not, whatever it derives from.
    assert schema_for(PeopleGraph) == (False, None)
    assert schema_for(_OtherGraph) == (False, None)
    assert schema_for(PeopleGraph(people=[])) == (False, None)


def test_a_graph_model_whose_schema_names_what_it_does_not_define_stays_here(monkeypatch):
    """A ``$ref`` to a class the schema does not define cannot be built elsewhere."""
    from cognee.shared import graph_model_utils

    graph_model = _people_graph_from_its_schema()
    monkeypatch.setattr(
        graph_model_utils,
        "graph_model_to_graph_schema",
        lambda model: {
            "title": "PeopleGraph",
            "properties": {"people": {"items": {"$ref": "#/$defs/Missing"}}},
            "$defs": {"Person": {"type": "object"}},
        },
    )

    assert cognify_module.graph_model_schema_for_workers(graph_model) == (False, None)


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
        {"graph_model": FROM_ITS_SCHEMA, "chunk_attachment": "direct"},
    ],
    ids=str,
)
async def test_another_process_rebuilds_the_standard_task_list_the_owner_built(cognify_kwargs):
    """With the real task builders on both sides: the owner builds from the
    arguments as given, the worker from what the owner resolved, and the two
    lists must do the same thing. Real builders, because with stand-ins this
    compares two constants and passes while the real lists differ."""
    if cognify_kwargs.get("graph_model") == FROM_ITS_SCHEMA:
        cognify_kwargs = {**cognify_kwargs, "graph_model": _people_graph_from_its_schema()}
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

    for route in (CognifyRoute.STANDARD, CognifyRoute.DLT_SOURCE):
        assert _described(worker.tasks_by_route[route]) == _described(owner.tasks_by_route[route])


@pytest.mark.asyncio
async def test_another_process_knows_the_standard_and_the_dlt_route():
    """A file and a DLT source are each one item another process can run whole,
    so a dataset of many sources is shared out like its files. A code repository
    is a directory on the owner's disk, so the code routes are not rebuilt: an
    item of theirs sent to another process fails there by name instead of
    running a list that differs from the owner's."""
    resolver, _ = await _resolver_of(chunks_per_batch=7)
    _, args = resolver.portable

    dlt_tasks = AsyncMock(return_value="DLT")
    with (
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value="STANDARD")),
        patch.object(cognify_module, "get_dlt_tasks", new=dlt_tasks),
        patch.object(cognify_module, "ensure_extractor_runtime", new=AsyncMock()),
        patch.object(
            cognify_module, "get_configured_ontology_resolver", new=MagicMock(return_value=None)
        ),
    ):
        rebuilt = await cognify_module.build_portable_cognify_task_resolver(user=object(), **args)

    assert rebuilt.tasks_by_route == {
        CognifyRoute.STANDARD: "STANDARD",
        CognifyRoute.DLT_SOURCE: "DLT",
    }
    # With the batch the owner's DLT list was built with.
    assert dlt_tasks.await_args.kwargs["chunks_per_batch"] == 7
    assert set(resolver.tasks_by_route) == set(CognifyRoute)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", list(CognifyRoute))
async def test_only_items_the_rebuilt_resolver_knows_are_offered_to_another_process(route):
    """The owner's resolver says which items the rebuilt one can run, so an
    executor never sends an item to a process that would fail it by name: files
    and DLT sources for the cognify run, files alone for the indexing run, which
    has no work for a DLT source."""
    resolver, _ = await _resolver_of()
    indexing = await _indexing_resolver_of(early_chunk_indexing=True)

    with patch.object(cognify_module, "cognify_route_for", return_value=route):
        assert resolver.portable_item(object()) is (
            route in (CognifyRoute.STANDARD, CognifyRoute.DLT_SOURCE)
        )
        assert indexing.portable_item(object()) is (route is CognifyRoute.STANDARD)


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
    # It names a reason exactly when it stays, and only one of its own.
    assert (resolver.stays_because is None) is portable


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
