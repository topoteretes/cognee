"""EMBEDDING_PROVIDER=embedding_worker end to end, against a broker and a real worker.

Skipped unless a worker is configured, e.g. for the worker's bge-small model::

    EMBEDDING_PROVIDER=embedding_worker EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \\
    EMBEDDING_DIMENSIONS=384 EMBEDDING_MAX_COMPLETION_TOKENS=512 \\
    COGNEE_EMBEDDING_WORKER_ENDPOINT=amqp://guest:guest@localhost:5672/ \\
        pytest cognee/tests/e2e/embedding_worker

with the worker started as ``EMBEDDING_WORKER_TRANSPORT=rabbitmq
EMBEDDING_WORKER_RABBITMQ_URL=amqp://guest:guest@localhost:5672/
EMBEDDING_WORKER_MODEL=bge-small-en-v1.5 embedding-worker``. No LLM key is needed:
the pipeline test chunks and indexes (``get_indexing_tasks``) and searches
``CHUNKS``, none of which call an LLM. A run that reaches the end prints
``EMBEDDING-WORKER-E2E-RAN``.
"""

from __future__ import annotations

import math
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EMBEDDING_PROVIDER") != "embedding_worker"
    or not os.environ.get("COGNEE_EMBEDDING_WORKER_ENDPOINT"),
    reason="EMBEDDING_PROVIDER=embedding_worker / COGNEE_EMBEDDING_WORKER_ENDPOINT are not set",
)

DOCUMENTS = [
    "Alice Novak moved to Lisbon and works as an architect at a small design studio.",
    "The Rust compiler checks ownership and borrowing at compile time.",
    "Bananas are rich in potassium and are a popular breakfast fruit.",
]


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


@pytest.mark.asyncio
async def test_the_engine_embeds_through_the_worker():
    from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (
        EmbeddingWorkerModelMismatchError,
        close_embedding_worker_transports,
    )
    from cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine import (
        EmbeddingWorkerEmbeddingEngine,
    )
    from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
        get_embedding_engine,
    )

    engine = get_embedding_engine()
    assert isinstance(engine, EmbeddingWorkerEmbeddingEngine)
    dimensions = engine.get_vector_size()

    texts = [*DOCUMENTS, "Where does Alice live?", "   "]
    result = await engine.embed_text(texts)
    assert len(result) == len(texts)
    assert all(len(vector) == dimensions for vector in result)
    assert result[-1] == [0.0] * dimensions  # blank input
    assert all(abs(sum(x * x for x in vector) - 1.0) < 1e-3 for vector in result[:-1])
    assert result.tokens > 0
    query = result[3]
    assert cosine(query, result[0]) > max(cosine(query, result[1]), cosine(query, result[2]))
    assert engine.served_model

    # Far past any small model's window: the worker answers input_too_long and
    # the engine splits and pools instead of failing.
    long_text = " ".join(DOCUMENTS * 200)
    [pooled] = await engine.embed_text([long_text])
    assert len(pooled) == dimensions

    # A worker serving another model than the expected one fails, and stays failed.
    mismatched = EmbeddingWorkerEmbeddingEngine(
        model=engine.model,
        dimensions=dimensions,
        settings=engine.settings.model_copy(update={"expected_model": "not/the-model"}),
    )
    for _ in range(2):
        with pytest.raises(EmbeddingWorkerModelMismatchError):
            await mismatched.embed_text(["x"])

    await close_embedding_worker_transports()


@pytest.mark.asyncio
async def test_add_index_and_search_chunks(tmp_path):
    import cognee
    from cognee.api.v1.cognify.cognify import get_indexing_tasks
    from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (
        close_embedding_worker_transports,
    )
    from cognee.modules.search.types import SearchType

    os.environ.update(TELEMETRY_DISABLED="1")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))

    await cognee.add(DOCUMENTS, dataset_name="embedding_worker_e2e")
    await cognee.run_custom_pipeline(
        tasks=await get_indexing_tasks(),
        dataset="embedding_worker_e2e",
        pipeline_name="indexing_pipeline",
    )
    results = await cognee.search(
        "Which programming language checks borrowing?",
        query_type=SearchType.CHUNKS,
        datasets=["embedding_worker_e2e"],
        top_k=1,
    )
    assert "Rust" in str(results)

    await close_embedding_worker_transports()
    print("EMBEDDING-WORKER-E2E-RAN")
