"""A dataset answers chunk searches before its entities are extracted (SDK-397).

Runs the real default pipeline on the default stack (sqlite + LanceDB + Ladybug)
with a mocked LLM and mocked embeddings, and proves the two things the early
chunk write promises: a CHUNKS search answers while ``extract_graph_and_summarize``
has not run yet, and no chunk is embedded twice.
"""

import importlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest

MARKER = re.compile(r"\bENT[A-Z]+\b")
DATASET = "early_chunk_indexing_test"
TEXT = (
    "Paragraph alpha ENTALPHA. Alice followed the white rabbit down the rabbit hole "
    "and found a golden key.\n\n"
    "Paragraph beta ENTBETA. The Queen of Hearts ruled the garden at the bottom "
    "of the well, and the Cheshire Cat grinned from a tree."
)


@pytest.fixture(scope="module")
def early_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_early_chunk_test_"))

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


@pytest.mark.asyncio
async def test_chunks_answer_before_extraction_and_are_embedded_once(early_env, monkeypatch):
    import cognee
    from cognee.modules.search.types import SearchType

    # The packages export same-named functions, so attribute access would return those.
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    adp_module = importlib.import_module("cognee.tasks.storage.add_data_points")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    async def chunk_search():
        # MOCK_EMBEDDING vectors are all zeros, so a vector search cannot rank; the
        # lexical search reads the chunk nodes the early write stored in the graph.
        return await cognee.search(
            query_text="rabbit hole",
            query_type=SearchType.CHUNKS_LEXICAL,
            datasets=[DATASET],
            top_k=5,
        )

    async def chunk_vector_rows() -> int:
        """Rows in the chunk-text collection: the early write's vector side."""
        from cognee.infrastructure.databases.vector import get_vector_engine_async

        connection = await (await get_vector_engine_async()).get_connection()
        if "DocumentChunk_text" not in await connection.table_names():
            return 0
        return int(await (await connection.open_table("DocumentChunk_text")).count_rows())

    before_extraction: list = []
    vector_rows_before_extraction: list[int] = []
    original_extract = cognify_module.extract_graph_and_summarize

    async def probing_extract(data_chunks, *args, **kwargs):
        # Between the early write and the extraction: the dataset must already answer.
        before_extraction.extend(await chunk_search())
        vector_rows_before_extraction.append(await chunk_vector_rows())
        return await original_extract(data_chunks, *args, **kwargs)

    monkeypatch.setattr(cognify_module, "extract_graph_and_summarize", probing_extract)

    embedded_chunk_ids: list[str] = []
    original_index = adp_module.index_data_points

    async def counting_index(data_points, *args, **kwargs):
        embedded_chunk_ids.extend(
            str(point.id) for point in data_points if type(point).__name__ == "DocumentChunk"
        )
        return await original_index(data_points, *args, **kwargs)

    monkeypatch.setattr(adp_module, "index_data_points", counting_index)

    await cognee.add(TEXT, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])

    assert before_extraction, "chunk search answered nothing before extraction ran"
    assert "rabbit hole" in json.dumps(before_extraction, default=str), (
        f"the early write did not surface the chunk text: {before_extraction!r}"
    )
    assert vector_rows_before_extraction and vector_rows_before_extraction[0] > 0, (
        "no chunk vector existed before extraction ran"
    )

    after = await chunk_search()
    assert "rabbit hole" in json.dumps(after, default=str)
    # The final write still attaches the extracted entities to the early-written
    # chunks (zero mock vectors cannot rank, so read the graph instead of searching it).
    from cognee.infrastructure.databases.graph import get_graph_engine

    entity_nodes, _ = await (await get_graph_engine()).get_filtered_graph_data(
        [{"type": ["Entity"]}]
    )
    entity_names = {str(props.get("name")).lower() for _, props in entity_nodes}
    assert {"entalpha", "entbeta"} <= entity_names, (
        f"extraction output did not reach the graph: {sorted(entity_names)}"
    )

    assert embedded_chunk_ids, "no chunk reached the embedder"
    assert len(embedded_chunk_ids) == len(set(embedded_chunk_ids)), (
        f"a chunk was embedded twice: {embedded_chunk_ids}"
    )
