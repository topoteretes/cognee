import pytest
import pathlib
import pytest_asyncio
import cognee

from cognee.low_level import setup as setup_databases
from cognee.tasks.storage import add_data_points
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever


ALPHA_TEXT = "orion orion logistics common"
BETA_TEXT = "orion logistics logistics logistics common"
GAMMA_TEXT = "nebula archive common"


def _clear_engine_caches():
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()


@pytest_asyncio.fixture
async def setup_bm25_corpus():
    """Persist a tiny deterministic corpus as DocumentChunk graph nodes (no cognify/LLM)."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    cognee.config.system_root_directory(str(base_dir / ".cognee_system/test_bm25_retriever"))
    cognee.config.data_root_directory(str(base_dir / ".data_storage/test_bm25_retriever"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    _clear_engine_caches()
    await setup_databases()

    document = TextDocument(
        name="Logistics notes",
        raw_data_location="somewhere",
        external_metadata="",
        mime_type="text/plain",
    )
    chunks = [
        DocumentChunk(
            text=text,
            chunk_size=len(text.split()),
            chunk_index=index,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
        )
        for index, text in enumerate([ALPHA_TEXT, BETA_TEXT, GAMMA_TEXT])
    ]
    await add_data_points(chunks)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        _clear_engine_caches()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_bm25_ranks_repeated_term_chunk_first(setup_bm25_corpus):
    """BM25 ranks the chunk with the most query-term occurrences highest, end to end."""
    retriever = BM25ChunksRetriever(top_k=4)

    chunks = await retriever.get_retrieved_objects("logistics")
    texts = [chunk["text"] for chunk in chunks]

    # beta repeats "logistics" three times, alpha once, gamma not at all.
    # Only the matched chunks have a defined order; gamma ties with any other zero-score chunk.
    assert texts[0] == BETA_TEXT
    assert texts[1] == ALPHA_TEXT
