"""A dataset becomes searchable in stages while cognify() runs (SDK-397).

Runs the real three-stage build on the default stack (sqlite + LanceDB +
Ladybug) with a mocked LLM and mocked embeddings and proves what each stage
promises: after the caching run the documents answer ``DOCUMENTS_LEXICAL``,
after the indexing run the chunks answer ``CHUNKS_LEXICAL`` while
``extract_graph_and_summarize`` has not run yet, the memory state moves
``cached`` -> ``indexed`` -> ``ready``, and no chunk is embedded twice.
"""

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
        "DOCUMENT_CACHING": "true",
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
async def test_memory_answers_more_at_each_stage_and_embeds_once(staged_env, monkeypatch):
    import cognee
    from cognee.modules.search.types import SearchType

    # The packages export same-named functions, so attribute access would return those.
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    adp_module = importlib.import_module("cognee.tasks.storage.add_data_points")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # --- probe 1: inside the indexing run, i.e. after the caching run completed.
    during_indexing: dict = {}
    original_chunking = cognify_module.extract_chunks_from_documents

    async def probing_chunking(documents, *args, **kwargs):
        if "state" not in during_indexing:
            during_indexing["state"] = await _memory_state()
            during_indexing["documents"] = await _search(
                SearchType.DOCUMENTS_LEXICAL, "rabbit hole"
            )
        async for chunk in original_chunking(documents, *args, **kwargs):
            yield chunk

    # --- probe 2: between the indexing run and extraction.
    before_extraction: dict = {}
    original_extract = cognify_module.extract_graph_and_summarize

    async def probing_extract(data_chunks, *args, **kwargs):
        if "state" not in before_extraction:
            before_extraction["state"] = await _memory_state()
            before_extraction["chunks"] = await _search(SearchType.CHUNKS_LEXICAL)
        return await original_extract(data_chunks, *args, **kwargs)

    embedded_chunk_ids: list[str] = []
    original_index = adp_module.index_data_points

    async def counting_index(data_points, *args, **kwargs):
        embedded_chunk_ids.extend(
            str(point.id) for point in data_points if type(point).__name__ == "DocumentChunk"
        )
        return await original_index(data_points, *args, **kwargs)

    monkeypatch.setattr(cognify_module, "extract_chunks_from_documents", probing_chunking)
    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", probing_extract)
    monkeypatch.setattr(adp_module, "index_data_points", counting_index)

    await cognee.add(TEXT, dataset_name=DATASET)
    result = await cognee.cognify(datasets=[DATASET])

    # Stage 1 done when stage 2 starts: cached, and the documents answer by keyword.
    assert during_indexing["state"][0] in ("indexing", "cached"), during_indexing["state"]
    assert during_indexing["state"][1] == "documents"
    documents_hit = json.dumps(during_indexing["documents"], default=str)
    assert "rabbit hole" in documents_hit, f"DOCUMENTS_LEXICAL did not answer: {documents_hit}"

    # Stage 2 done when extraction starts: indexed, and the chunks answer.
    assert before_extraction["state"][0] in ("extracting", "indexed"), before_extraction["state"]
    assert before_extraction["state"][1] == "chunks"
    assert "rabbit hole" in json.dumps(before_extraction["chunks"], default=str)

    # Stage 3 done: ready, entities in the graph, run info stamped.
    assert await _memory_state() == ("ready", "graph")
    run_info = next(iter(result.values()))
    assert run_info.memory_state == "ready"
    from cognee.infrastructure.databases.graph import get_graph_engine

    entity_nodes, _ = await (await get_graph_engine()).get_filtered_graph_data(
        [{"type": ["Entity"]}]
    )
    entity_names = {str(props.get("name")).lower() for _, props in entity_nodes}
    assert {"entalpha", "entbeta"} <= entity_names, sorted(entity_names)
    cached_nodes, _ = await (await get_graph_engine()).get_filtered_graph_data(
        [{"type": ["CachedDocument"]}]
    )
    assert len(cached_nodes) == 1, "one cached document per document"

    assert embedded_chunk_ids, "no chunk reached the embedder"
    assert len(embedded_chunk_ids) == len(set(embedded_chunk_ids)), (
        f"a chunk was embedded twice: {embedded_chunk_ids}"
    )

    # Every stage left its run record and its stamp.
    dataset_id = await _dataset_id()
    statuses = await cognee.datasets.get_status(
        [dataset_id], ["caching_pipeline", "indexing_pipeline", "cognify_pipeline"]
    )
    per_pipeline = statuses[str(dataset_id)]
    assert {name: str(getattr(s, "value", s)) for name, s in per_pipeline.items()} == {
        "caching_pipeline": "DATASET_PROCESSING_COMPLETED",
        "indexing_pipeline": "DATASET_PROCESSING_COMPLETED",
        "cognify_pipeline": "DATASET_PROCESSING_COMPLETED",
    }

    # The full memory answers DOCUMENTS_LEXICAL too, and a second cognify is a no-op.
    assert "rabbit hole" in json.dumps(await _search(SearchType.DOCUMENTS_LEXICAL), default=str)
    before = len(embedded_chunk_ids)
    await cognee.cognify(datasets=[DATASET])
    assert len(embedded_chunk_ids) == before, "a re-run re-embedded chunks"
    assert await _memory_state() == ("ready", "graph")
