"""A dataset answers chunk searches before its entities are extracted (SDK-397).

Runs the real two-run build (indexing, then cognify) on the default stack
(sqlite + LanceDB + Ladybug) with a mocked LLM and mocked embeddings and proves
what the indexing run promises: the chunks answer ``CHUNKS`` and
``CHUNKS_LEXICAL`` while ``extract_graph_and_summarize`` has not run yet,
recall answers from them and says the graph is not built, the memory state
moves ``extracting`` (at chunk level) -> ``ready``, and no chunk is embedded
twice.
"""

import asyncio
import importlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest

MARKER = re.compile(r"\bENT[A-Z]+\b")
DATASET = "staged_memory_build_test"
TEXT = (
    "Paragraph alpha ENTALPHA. Alice followed the white rabbit down the rabbit hole "
    "and found a golden key.\n\n"
    "Paragraph beta ENTBETA. The Queen of Hearts ruled the garden at the bottom "
    "of the well, and the Cheshire Cat grinned from a tree."
)


@pytest.fixture(scope="module")
def event_loop():
    """One loop for the module: the engines its tests share hold locks bound to it."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def staged_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_staged_memory_test_"))

    import cognee

    def clear_config_caches():
        for module_name, factory_name in [
            ("cognee.base_config", "get_base_config"),
            ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
            (
                "cognee.infrastructure.databases.relational.get_relational_engine",
                "get_relational_engine",
            ),
            ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
            ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
            ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
            ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
            ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
            (
                "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
                "create_embedding_engine",
            ),
            ("cognee.infrastructure.llm.config", "get_llm_config"),
            ("cognee.modules.cognify.config", "get_cognify_config"),
        ]:
            try:
                getattr(importlib.import_module(module_name), factory_name).cache_clear()
            except (ImportError, AttributeError):
                pass

    mp = pytest.MonkeyPatch()
    for key, value in {
        "DB_PROVIDER": "sqlite",
        "VECTOR_DB_PROVIDER": "lancedb",
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "CACHE_BACKEND": "sqlite",
        "MOCK_EMBEDDING": "true",
        "TELEMETRY_DISABLED": "1",
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "EARLY_CHUNK_INDEXING": "true",
    }.items():
        mp.setenv(key, value)
    clear_config_caches()

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(MARKER.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=n) for n in names], edges=[]
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    yield root

    LLMGateway.acreate_structured_output = original
    mp.undo()
    clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


async def _dataset_id():
    import cognee

    for dataset in await cognee.datasets.list_datasets():
        if dataset.name == DATASET:
            return dataset.id
    return None


async def _memory_state():
    import cognee

    dataset_id = await _dataset_id()
    status = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
    return status.state, status.search_level


async def _search(query_type, query="rabbit hole"):
    import cognee

    return await cognee.search(query_text=query, query_type=query_type, datasets=[DATASET], top_k=5)


@pytest.mark.asyncio
async def test_chunks_answer_before_extraction_and_are_embedded_once(staged_env, monkeypatch):
    import cognee
    from cognee.modules.search.types import SearchType

    # The packages export same-named functions, so attribute access would return those.
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    adp_module = importlib.import_module("cognee.tasks.storage.add_data_points")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # The probe: after the indexing run, before extraction.
    before_extraction: dict = {}
    original_extract = cognify_module.extract_graph_and_summarize

    async def probing_extract(data_chunks, *args, **kwargs):
        if "state" not in before_extraction:
            before_extraction["state"] = await _memory_state()
            # Keyword search: the mocked embeddings cannot rank a vector search.
            before_extraction["chunks"] = await _search(SearchType.CHUNKS_LEXICAL)
            before_extraction["recall"] = await cognee.recall(
                "rabbit hole",
                query_type=SearchType.CHUNKS_LEXICAL,
                datasets=[DATASET],
                top_k=5,
            )
            # What the indexing run embedded: the cognify run must add nothing.
            before_extraction["embedded"] = sorted(embedded_chunk_ids)
        return await original_extract(data_chunks, *args, **kwargs)

    embedded_chunk_ids: list[str] = []
    original_index = adp_module.index_data_points

    async def counting_index(data_points, *args, **kwargs):
        embedded_chunk_ids.extend(
            str(point.id) for point in data_points if type(point).__name__ == "DocumentChunk"
        )
        return await original_index(data_points, *args, **kwargs)

    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", probing_extract)
    monkeypatch.setattr(adp_module, "index_data_points", counting_index)

    await cognee.add(TEXT, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])

    # Indexed when extraction starts: the chunks answer.
    assert before_extraction["state"] == ("extracting", "chunks"), before_extraction["state"]
    assert "rabbit hole" in json.dumps(before_extraction["chunks"], default=str)
    # Recall answers from the chunks and says the graph is still being built.
    recalled = before_extraction["recall"]
    markers = [entry for entry in recalled if getattr(entry, "source", None) == "system"]
    assert [marker.status for marker in markers] == ["memory_extracting"], recalled
    assert "chunk-level search" in markers[0].text
    assert "rabbit hole" in json.dumps([e.model_dump() for e in recalled], default=str)

    # Extraction done: ready, entities in the graph.
    assert await _memory_state() == ("ready", "graph")
    from cognee.infrastructure.databases.graph import get_graph_engine

    entity_nodes, _ = await (await get_graph_engine()).get_filtered_graph_data(
        [{"type": ["Entity"]}]
    )
    entity_names = {str(props.get("name")).lower() for _, props in entity_nodes}
    assert {"entalpha", "entbeta"} <= entity_names, sorted(entity_names)

    assert embedded_chunk_ids, "no chunk reached the embedder"
    assert len(embedded_chunk_ids) == len(set(embedded_chunk_ids)), (
        f"a chunk was embedded twice: {embedded_chunk_ids}"
    )
    # Every chunk was embedded by the indexing run, before extraction started.
    assert before_extraction["embedded"] == sorted(embedded_chunk_ids), (
        "the cognify run embedded chunks the indexing run should have"
    )
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    chunk_nodes, _ = await (await get_graph_engine()).get_filtered_graph_data(
        [{"type": ["DocumentChunk"]}]
    )
    chunk_ids = [str(node_id) for node_id, _ in chunk_nodes]
    rows = await (await get_vector_engine_async()).retrieve("DocumentChunk_text", chunk_ids)
    assert sorted(str(row.id) for row in rows) == sorted(chunk_ids), "a chunk has no vector row"

    # Both runs left their run record.
    dataset_id = await _dataset_id()
    statuses = await cognee.datasets.get_status(
        [dataset_id], ["indexing_pipeline", "cognify_pipeline"]
    )
    per_pipeline = statuses[str(dataset_id)]
    assert {name: str(getattr(s, "value", s)) for name, s in per_pipeline.items()} == {
        "indexing_pipeline": "DATASET_PROCESSING_COMPLETED",
        "cognify_pipeline": "DATASET_PROCESSING_COMPLETED",
    }

    # A second cognify is a no-op: nothing indexed or embedded again.
    before = len(embedded_chunk_ids)
    await cognee.cognify(datasets=[DATASET])
    assert len(embedded_chunk_ids) == before, "a re-run re-embedded chunks"
    assert await _memory_state() == ("ready", "graph")


@pytest.mark.asyncio
async def test_a_failed_extraction_leaves_the_chunks_searchable(staged_env, monkeypatch):
    import cognee
    from cognee.modules.search.types import SearchType

    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    dataset = "staged_failed_extraction"
    await cognee.add(
        "The lighthouse keeper logged every passing ship in a leather notebook.",
        dataset_name=dataset,
    )

    async def failing_extract(*args, **kwargs):
        raise RuntimeError("injected extraction outage")

    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", failing_extract)
    with pytest.raises(Exception, match="injected extraction outage"):
        await cognee.cognify(datasets=[dataset])

    dataset_id = next(d.id for d in await cognee.datasets.list_datasets() if d.name == dataset)
    status = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
    assert (status.state, status.search_level) == ("extraction_failed", "chunks")
    hits = await cognee.search(
        query_text="leather notebook",
        query_type=SearchType.CHUNKS_LEXICAL,
        datasets=[dataset],
        top_k=3,
    )
    assert "leather notebook" in json.dumps(hits, default=str)


@pytest.mark.asyncio
async def test_a_retry_with_another_chunk_size_leaves_one_set_of_chunks(staged_env, monkeypatch):
    """The indexing run's chunks outlive a failed extraction; a retry that cuts
    the document differently must not keep them as a second copy of the text."""
    import cognee
    from cognee.infrastructure.databases.graph import get_graph_engine

    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    dataset = "staged_retry_chunk_size"
    text = " ".join(f"Sentence {i} about the retry harbour lighthouse." for i in range(60))

    async def chunk_tokens():
        nodes, _ = await (await get_graph_engine()).get_filtered_graph_data(
            [{"type": ["DocumentChunk"]}]
        )
        # The graph is shared with the other test's dataset: this document's chunks only.
        ours = [props for _, props in nodes if "retry harbour" in (props.get("text") or "")]
        return sorted({props.get("max_chunk_tokens") for props in ours}), len(ours)

    await cognee.add(text, dataset_name=dataset)

    original_extract = cognify_module.extract_graph_and_summarize

    async def failing_extract(*args, **kwargs):
        raise RuntimeError("injected extraction outage")

    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", failing_extract)
    with pytest.raises(Exception, match="injected extraction outage"):
        await cognee.cognify(datasets=[dataset], chunk_size=64)
    small_tokens, small_count = await chunk_tokens()
    assert small_count > 0, "the indexing run should have stored the first cut"

    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", original_extract)
    await cognee.cognify(datasets=[dataset], chunk_size=256)

    tokens, count = await chunk_tokens()
    assert len(tokens) == 1 and tokens != small_tokens, (
        f"chunks of both cuts are stored: {tokens} (first cut {small_tokens})"
    )
    assert count < small_count


@pytest.mark.asyncio
async def test_a_background_build_hands_back_the_cognify_run(staged_env):
    """Callers follow the cognify run, as before: its id comes back at once, its row
    reads as running from the start, and the memory state follows both runs."""
    import cognee
    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunStarted

    dataset = "staged_background_build"
    await cognee.add("The ferry to the island leaves at dawn every Tuesday.", dataset_name=dataset)

    started = await cognee.cognify(datasets=[dataset], run_in_background=True)

    ((dataset_id, run_info),) = started.items()
    assert isinstance(run_info, PipelineRunStarted)
    statuses = await cognee.datasets.get_status([dataset_id], ["cognify_pipeline"])
    run_status = statuses[str(dataset_id)]
    assert str(getattr(run_status, "value", run_status)) in (
        "DATASET_PROCESSING_STARTED",
        "DATASET_PROCESSING_COMPLETED",
    )

    for _ in range(200):
        status = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
        if status.state == "ready":
            break
        assert status.state in ("indexing", "extracting"), status
        await asyncio.sleep(0.1)
    assert (status.state, status.search_level) == ("ready", "graph")

    from cognee.modules.pipelines.methods.get_pipeline_run import get_latest_pipeline_run

    run = await get_latest_pipeline_run(run_info.pipeline_run_id)
    assert run.pipeline_name == "cognify_pipeline"


@pytest.mark.asyncio
async def test_a_partly_failed_indexing_run_keeps_its_completed_documents(staged_env, monkeypatch):
    """One document fails indexing, the other is kept (per-item rollback); then
    extraction fails. The kept chunks must still count as searchable, recall must
    answer from them, and once the build succeeds the kept document must be a
    baseline for update(), not a full rebuild."""
    import cognee
    from cognee.modules.data.methods import get_dataset_data
    from cognee.modules.search.types import SearchType

    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    dataset = "staged_partial_indexing"
    # Larger first: the run processes items largest first, one at a time here.
    kept_text = " ".join(["The orchard cooperative pressed cider every October."] * 20)
    await cognee.add(kept_text, dataset_name=dataset)
    await cognee.add("FAILINDEX: the bakery opens at six.", dataset_name=dataset)

    original_index = cognify_module.index_chunks_for_search

    async def failing_index(data_chunks, ctx=None):
        if any("FAILINDEX" in chunk.text for chunk in data_chunks):
            raise RuntimeError("injected indexing outage")
        return await original_index(data_chunks, ctx=ctx)

    async def failing_extract(*args, **kwargs):
        raise RuntimeError("injected extraction outage")

    original_extract = cognify_module.extract_graph_and_summarize
    # Failed items are recorded and the run carries on, so extraction runs; by
    # default the failed item would raise and fail cognify at once.
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "false")
    monkeypatch.setattr(cognify_module, "index_chunks_for_search", failing_index)
    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", failing_extract)
    with pytest.raises(Exception, match="injected extraction outage"):
        await cognee.cognify(datasets=[dataset], data_per_batch=1)

    dataset_id = next(d.id for d in await cognee.datasets.list_datasets() if d.name == dataset)
    statuses = await cognee.datasets.get_status([dataset_id], ["indexing_pipeline"])
    indexing = statuses[str(dataset_id)]
    assert str(getattr(indexing, "value", indexing)) == "DATASET_PROCESSING_ERRORED"
    status = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
    assert (status.state, status.search_level) == ("extraction_failed", "chunks"), status

    recalled = await cognee.recall(
        "orchard cider", query_type=SearchType.CHUNKS_LEXICAL, datasets=[dataset], top_k=3
    )
    markers = [entry for entry in recalled if getattr(entry, "source", None) == "system"]
    assert [marker.status for marker in markers] == ["memory_extraction_failed"], recalled
    assert "orchard" in json.dumps([e.model_dump() for e in recalled], default=str)

    monkeypatch.setattr(cognify_module, "index_chunks_for_search", original_index)
    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", original_extract)
    monkeypatch.delenv("RAISE_INCREMENTAL_LOADING_ERRORS")
    await cognee.cognify(datasets=[dataset])
    status = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
    assert status.state == "ready", status

    kept = next(item for item in await get_dataset_data(dataset_id) if "FAILINDEX" not in item.name)
    result = await cognee.update(data_id=kept.id, data=kept_text, dataset_id=dataset_id)
    assert result["status"] == "unchanged", result


async def _indexing_run_count(dataset_id):
    from sqlalchemy import func, select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.pipelines.models import PipelineRun

    async with get_relational_engine().get_async_session() as session:
        return await session.scalar(
            select(func.count(func.distinct(PipelineRun.pipeline_run_id))).where(
                PipelineRun.dataset_id == dataset_id,
                PipelineRun.pipeline_name == "indexing_pipeline",
            )
        )


@pytest.mark.asyncio
async def test_a_memory_only_forget_of_one_document_indexes_it_early_again(staged_env):
    """The forget drops the document's chunks, so its indexing stamp goes too: the
    next cognify indexes it before extraction instead of skipping it."""
    import cognee

    dataset = "staged_forget_memory_only"
    await cognee.add("The night train to Vienna stops at every frozen lake.", dataset_name=dataset)
    await cognee.cognify(datasets=[dataset])
    dataset_id = next(d.id for d in await cognee.datasets.list_datasets() if d.name == dataset)
    (data,) = await cognee.datasets.list_data(dataset_id)
    assert await _indexing_run_count(dataset_id) == 1

    await cognee.forget(data_id=data.id, dataset_id=dataset_id, memory_only=True)

    (data,) = await cognee.datasets.list_data(dataset_id)
    stamped = {name for name, by_dataset in data.pipeline_status.items() if by_dataset}
    assert "indexing_pipeline" not in stamped and "cognify_pipeline" not in stamped

    await cognee.cognify(datasets=[dataset])

    assert await _indexing_run_count(dataset_id) == 2
    status = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
    assert (status.state, status.search_level) == ("ready", "graph")


@pytest.mark.asyncio
async def test_a_run_level_indexing_failure_fails_cognify_at_once(staged_env, monkeypatch):
    """Not item-specific, so extraction would fail the same way after re-reading the
    dataset: it never starts, and the cognify run records the indexing cause."""
    import cognee

    memory_pipelines = importlib.import_module("cognee.modules.cognify.memory_pipelines")
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    dataset = "staged_indexing_outage"
    await cognee.add("The orchard keeper counts the pears every September.", dataset_name=dataset)

    original_run_pipeline = memory_pipelines.run_pipeline

    def run_pipeline(**kwargs):
        if kwargs["pipeline_name"] != "indexing_pipeline":
            return original_run_pipeline(**kwargs)

        async def unreachable():
            raise ConnectionError("injected vector store outage")
            yield

        return unreachable()

    extracted = []

    async def tracking_extract(*args, **kwargs):
        extracted.append(True)
        raise AssertionError("extraction must not start")

    monkeypatch.setattr(memory_pipelines, "run_pipeline", run_pipeline)
    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", tracking_extract)

    # raise_on_error=False keeps the raw cause; the default wraps it.
    with pytest.raises(ConnectionError, match="injected vector store outage"):
        await cognee.cognify(datasets=[dataset], raise_on_error=False)

    assert not extracted
    dataset_id = next(d.id for d in await cognee.datasets.list_datasets() if d.name == dataset)
    status = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
    assert status.state == "extraction_failed"
    assert status.error_class == "ConnectionError"
    assert "injected vector store outage" in (status.error_message or "")
