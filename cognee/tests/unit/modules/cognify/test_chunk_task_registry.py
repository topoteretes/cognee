"""Tasks plugins register to run right after chunking (CLO-397).

Every task list that chunks documents runs the registered tasks right after its
chunking task: cognify's LLM list, its GLiNER demo list and the indexing
pipeline's list (all built with ``build_chunking_tasks``), and the dlt route's
list, whose chunks are source rows. They run in registration order, and an empty
registry leaves the lists exactly as they were.
"""

import importlib
from types import FunctionType, SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify import register_chunk_task, unregister_chunk_task
from cognee.modules.cognify.chunk_task_registry import chunk_task_factories
from cognee.modules.cognify.chunking_tasks import (
    build_chunking_tasks,
    build_registered_chunk_tasks,
    resolve_chunks_per_batch,
)
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base
from cognee.modules.pipelines.tasks.task import Task, pipeline_needs_llm
from cognee.pipelines import Drop
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
gliner_tasks_module = importlib.import_module("cognee.tasks.graph.gliner_demo.tasks")
chunk_task_module = importlib.import_module("cognee.tasks.documents.extract_chunks_from_documents")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")
migrations_startup_module = importlib.import_module("cognee.modules.migrations.startup")


@pytest.fixture(autouse=True)
def isolated_registry():
    """Each test starts with no chunk tasks and leaves the registry as it found it."""
    saved = chunk_task_factories()
    for name, _ in saved:
        unregister_chunk_task(name)
    yield
    for name, _ in chunk_task_factories():
        unregister_chunk_task(name)
    for name, factory in saved:
        register_chunk_task(name, factory)


@pytest.fixture
def pinned_config(monkeypatch):
    """Early chunk indexing on, provenance off, so each list's shape is fixed."""
    config = get_cognify_config().model_copy(
        update={"early_chunk_indexing": True, "provenance_tracking": False}
    )
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: config)
    monkeypatch.setattr(gliner_tasks_module, "get_cognify_config", lambda: config)


async def gate_chunks(chunks):
    return chunks


async def tag_chunks(chunks):
    return None


def gate_factory(*, chunks_per_batch, **kwargs):
    return Task(gate_chunks, batch_size=chunks_per_batch, needs_llm=False)


def tag_factory(*, chunks_per_batch, **kwargs):
    return [Task(tag_chunks, batch_size=chunks_per_batch, enriches=True, needs_llm=False)]


async def default_list(chunks_per_batch=None):
    return await cognify_module.get_default_tasks(chunk_size=512, chunks_per_batch=chunks_per_batch)


async def indexing_list(chunks_per_batch=None):
    return await cognify_module.get_indexing_tasks(
        chunk_size=512, chunks_per_batch=chunks_per_batch
    )


async def gliner_list(chunks_per_batch=None):
    with patch.object(gliner_tasks_module, "require_gliner2"):
        return await gliner_tasks_module.get_gliner_demo_tasks(
            ["person"], chunk_size=512, chunks_per_batch=chunks_per_batch
        )


async def dlt_list(chunks_per_batch=None):
    return await cognify_module.get_dlt_tasks(chunk_size=512, chunks_per_batch=chunks_per_batch)


# Each list built with chunks_per_batch=7 as it was before the hook existed:
# (task, batch_size, needs_llm) in order.
LISTS = {
    "default": (
        default_list,
        [
            ("classify_documents", 1, False),
            ("extract_chunks_from_documents", 1, False),
            ("find_indexed_chunks", 7, False),
            ("extract_graph_and_summarize", 7, True),
            ("add_data_points", 7, False),
        ],
    ),
    "indexing": (
        indexing_list,
        [
            ("classify_documents", 1, False),
            ("extract_chunks_from_documents", 1, False),
            ("index_chunks_for_search", 7, False),
        ],
    ),
    "gliner_demo": (
        gliner_list,
        [
            ("classify_documents", 1, False),
            ("prepare_gliner_schema", 1, False),
            ("extract_chunks_from_documents", 1, False),
            ("find_indexed_chunks", 7, False),
            ("extract_graph_and_summarize_with_gliner", 7, False),
            ("add_data_points", 7, False),
        ],
    ),
    "dlt": (
        dlt_list,
        [
            ("classify_documents", 1, False),
            ("purge_stale_dlt_source_artifacts", 1, False),
            ("extract_chunks_from_documents", 1, False),
            ("add_data_points", 7, False),
            ("extract_dlt_source_edges", 1, False),
        ],
    ),
}


def _names(tasks):
    return [task.executable.__name__ for task in tasks]


def _profile(tasks):
    return [
        (task.executable.__name__, task.task_config["batch_size"], task.needs_llm) for task in tasks
    ]


def _comparable(value):
    """Plain values, classes and functions as they are; objects built per call by type."""
    if isinstance(value, dict):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool, type, FunctionType)):
        return value
    return type(value)


def _shape(tasks):
    """Everything about a task list the runner reads."""
    return [
        (
            task.executable,
            _comparable(task.default_params["args"]),
            _comparable(task.default_params["kwargs"]),
            task.task_config,
            task.needs_llm,
            task.enriches,
        )
        for task in tasks
    ]


def test_nothing_registered_builds_the_classify_and_chunk_pair_unchanged():
    tasks = build_chunking_tasks(TextChunker, 512, temporal_hints=False, chunks_per_batch=7)

    before = (
        Task(classify_documents, needs_llm=False),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=512,
            chunker=TextChunker,
            temporal_hints=False,
            needs_llm=False,
        ),
    )
    assert _shape(tasks) == _shape(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("list_name", sorted(LISTS))
async def test_an_empty_registry_leaves_every_list_unchanged(list_name, pinned_config):
    build, profile_before = LISTS[list_name]
    calls = []

    def recording_factory(**kwargs):
        calls.append(kwargs)
        return gate_factory(**kwargs)

    untouched = await build(7)
    register_chunk_task("gate", recording_factory)
    unregister_chunk_task("gate")
    after_unregister = await build(7)

    assert _profile(untouched) == profile_before
    assert _shape(after_unregister) == _shape(untouched)
    assert calls == [], "an unregistered factory is never called"


@pytest.mark.asyncio
@pytest.mark.parametrize("list_name", sorted(LISTS))
async def test_the_registered_task_runs_right_after_chunking(list_name, pinned_config):
    build, _ = LISTS[list_name]
    untouched = await build(7)
    register_chunk_task("gate", gate_factory)

    tasks = await build(7)

    chunk_position = _names(untouched).index("extract_chunks_from_documents")
    expected = _names(untouched)
    expected.insert(chunk_position + 1, "gate_chunks")
    assert _names(tasks) == expected
    # Batched like the tasks after it, so they receive the batches they did before,
    # and every other task of the list is exactly what it was.
    assert tasks[chunk_position + 1].task_config["batch_size"] == 7
    assert _shape(tasks[: chunk_position + 1] + tasks[chunk_position + 2 :]) == _shape(untouched)


@pytest.mark.asyncio
async def test_the_factory_gets_the_resolved_batch_size(pinned_config):
    received = []

    def factory(*, chunks_per_batch, **kwargs):
        received.append(chunks_per_batch)
        return []

    register_chunk_task("probe", factory)
    await default_list(5)
    await indexing_list()
    await dlt_list(3)

    assert received == [5, resolve_chunks_per_batch(), 3]


@pytest.mark.asyncio
async def test_tasks_run_in_registration_order_and_a_factory_may_return_several(pinned_config):
    def gate_and_tag(*, chunks_per_batch, **kwargs):
        return [gate_factory(chunks_per_batch=chunks_per_batch)] + tag_factory(
            chunks_per_batch=chunks_per_batch
        )

    register_chunk_task("tag", tag_factory)
    register_chunk_task("gate_and_tag", gate_and_tag)

    tasks = await indexing_list(7)

    assert _names(tasks) == [
        "classify_documents",
        "extract_chunks_from_documents",
        "tag_chunks",
        "gate_chunks",
        "tag_chunks",
        "index_chunks_for_search",
    ]


def test_registering_a_name_again_replaces_its_factory_in_place():
    register_chunk_task("first", gate_factory)
    register_chunk_task("second", tag_factory)
    register_chunk_task("first", tag_factory)

    assert chunk_task_factories() == [("first", tag_factory), ("second", tag_factory)]
    assert _names(build_registered_chunk_tasks(7)) == ["tag_chunks", "tag_chunks"]


def test_unregister_returns_the_factory_once():
    register_chunk_task("gate", gate_factory)

    assert unregister_chunk_task("gate") is gate_factory
    assert unregister_chunk_task("gate") is None
    assert build_registered_chunk_tasks(7) == []


def test_a_factory_returning_no_tasks_adds_nothing():
    register_chunk_task("off", lambda **kwargs: [])

    assert len(build_chunking_tasks(TextChunker, 512)) == 2


@pytest.mark.parametrize(
    "returned", [None, gate_chunks, [Task(gate_chunks), "not a task"], iter([Task(gate_chunks)])]
)
def test_a_factory_must_return_tasks(returned):
    register_chunk_task("broken", lambda **kwargs: returned)

    with pytest.raises(TypeError, match="'broken'"):
        build_chunking_tasks(TextChunker, 512)


def test_the_factory_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        register_chunk_task("gate", Task(gate_chunks))


@pytest.mark.asyncio
async def test_an_llm_free_task_keeps_the_gliner_route_llm_free(pinned_config):
    """The run probes the LLM only when some task needs it (``pipeline_needs_llm``)."""
    register_chunk_task("gate", gate_factory)
    assert pipeline_needs_llm(await gliner_list(7)) is False
    assert pipeline_needs_llm(await dlt_list(7)) is False

    register_chunk_task(
        "gate", lambda *, chunks_per_batch, **kwargs: Task(gate_chunks, batch_size=chunks_per_batch)
    )
    assert pipeline_needs_llm(await gliner_list(7)) is True
    assert pipeline_needs_llm(await dlt_list(7)) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_llm", [False, True])
async def test_cognify_probes_the_llm_for_a_registered_task_only_when_it_says_so(
    needs_llm, pinned_config, monkeypatch
):
    """cognify() hands the run the union of ``needs_llm`` over every route's list
    (GLiNER, dlt, code): an LLM-free registered task keeps the GLiNER run LLM-free."""
    executor = AsyncMock(return_value={})
    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: None)
    monkeypatch.setattr(cognify_module, "get_pipeline_executor", lambda **kwargs: executor)
    monkeypatch.setattr(cognify_module, "ensure_extractor_runtime", AsyncMock())
    monkeypatch.setattr(cognify_module, "get_configured_ontology_resolver", lambda *a, **k: None)
    monkeypatch.setattr(migrations_startup_module, "run_migrations_and_block", AsyncMock())
    monkeypatch.setattr(gliner_tasks_module, "require_gliner2", lambda: None)
    register_chunk_task(
        "gate",
        lambda *, chunks_per_batch, **kwargs: Task(
            gate_chunks, batch_size=chunks_per_batch, needs_llm=needs_llm
        ),
    )

    await cognify_module.cognify(extractor="gliner_demo", chunk_size=512)

    assert executor.await_args.kwargs["needs_llm"] is needs_llm


@pytest.mark.asyncio
async def test_the_runner_hands_the_tasks_their_batches_and_the_next_task_their_result(
    tmp_path, monkeypatch
):
    """The documented contract, through the real runner and the real chunker: the
    first registered task's batch_size sets the chunker's batches, Drop removes a
    batch, an enriches task that returns None passes its input on, the task after
    them receives exactly what the gate returned, and ctx is injected."""
    monkeypatch.setattr(chunk_task_module, "update_document_token_count", AsyncMock())
    calls, gate_batches, kept_batches, sink_batches = [], [], [], []

    async def screen_chunks(chunks, ctx):
        calls.append(("screen", ctx.pipeline_name))
        gate_batches.append([chunk.chunk_index for chunk in chunks])
        if len(gate_batches) == 1:
            return Drop
        kept = [chunk for chunk in chunks if chunk.chunk_index % 2 == 0]
        kept_batches.append(kept)
        return kept

    async def mark_chunks(chunks, ctx):
        # Returns nothing: as an enriches task, its input goes on to the next task.
        calls.append(("mark", ctx.pipeline_name))

    async def next_task(chunks):
        sink_batches.append(chunks)
        return chunks

    register_chunk_task(
        "screen",
        lambda *, chunks_per_batch, **kwargs: Task(
            screen_chunks, batch_size=chunks_per_batch, needs_llm=False
        ),
    )
    register_chunk_task(
        "mark",
        lambda *, chunks_per_batch, **kwargs: Task(
            mark_chunks, batch_size=chunks_per_batch, enriches=True, needs_llm=False
        ),
    )
    text = " ".join(f"Sentence {index} is plain. Another sentence {index}." for index in range(30))
    (tmp_path / "doc.txt").write_text(text)
    data = SimpleNamespace(
        id=uuid4(),
        name="doc",
        extension="txt",
        mime_type="text/plain",
        raw_data_location=str(tmp_path / "doc.txt"),
        external_metadata={},
        importance_weight=None,
        node_set=None,
        system_metadata=None,
    )
    tasks = [
        *build_chunking_tasks(TextChunker, 24, chunks_per_batch=4),
        Task(next_task, batch_size=4, needs_llm=False),
    ]
    user = SimpleNamespace(id=uuid4(), tenant_id=None, email=None)
    ctx = PipelineContext(user=user, pipeline_name="cognify_pipeline")

    async for _ in run_tasks_base(tasks, [data], user, ctx):
        pass

    # The chunker handed the gate every chunk, in order, in batches of its batch size.
    assert len(gate_batches) > 2
    assert all(len(batch) == 4 for batch in gate_batches[:-1]) and len(gate_batches[-1]) <= 4
    chunk_count = sum(len(batch) for batch in gate_batches)
    assert [index for batch in gate_batches for index in batch] == list(range(chunk_count))
    # The first batch was dropped whole; every other one reached the next task as
    # the gate returned it (the same list), through the task that marks in place.
    assert len(sink_batches) == len(gate_batches) - 1
    assert all(sink is kept for sink, kept in zip(sink_batches, kept_batches, strict=True))
    assert all(sink and all(chunk.chunk_index % 2 == 0 for chunk in sink) for sink in sink_batches)
    # Registration order, once per batch each task was handed, with the run's ctx.
    pipeline = "cognify_pipeline"
    assert calls == [("screen", pipeline)] + [("screen", pipeline), ("mark", pipeline)] * len(
        sink_batches
    )
