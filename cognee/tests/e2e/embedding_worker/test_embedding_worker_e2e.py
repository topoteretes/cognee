"""EMBEDDING_PROVIDER=embedding_worker end to end, against a broker and a real worker.

Skipped unless a worker is configured, e.g. for the worker's bge-small model with
a low token cap, so that splitting over-long inputs is easy to exercise::

    docker run -d --rm --name rabbit -p 5672:5672 rabbitmq:4-alpine
    EMBEDDING_WORKER_TRANSPORT=rabbitmq \\
    EMBEDDING_WORKER_RABBITMQ_URL=amqp://guest:guest@localhost:5672/%2f \\
    EMBEDDING_WORKER_MODEL=bge-small-en-v1.5 EMBEDDING_WORKER_MAX_INPUT_TOKENS=64 \\
        embedding-worker

    EMBEDDING_PROVIDER=embedding_worker EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \\
    COGNEE_EMBEDDING_WORKER_ENDPOINT=amqp://guest:guest@localhost:5672/%2f \\
        pytest cognee/tests/e2e/embedding_worker

Leave EMBEDDING_DIMENSIONS and EMBEDDING_MAX_COMPLETION_TOKENS unset to check
that they come from the worker's model info (the tests that need that skip
themselves otherwise). No LLM key is needed: the pipeline test chunks and
indexes (``get_indexing_tasks``) and searches ``CHUNKS``, none of which call an
LLM. A run that reaches the end prints ``EMBEDDING-WORKER-E2E-RAN``.
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

needs_unset_dimensions = pytest.mark.skipif(
    bool(os.environ.get("EMBEDDING_DIMENSIONS")),
    reason="EMBEDDING_DIMENSIONS is set; this checks the size taken from the worker",
)

DOCUMENTS = [
    "Alice Novak moved to Lisbon and works as an architect at a small design studio.",
    "The Rust compiler checks ownership and borrowing at compile time.",
    "Bananas are rich in potassium and are a popular breakfast fruit.",
]


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def fresh_engine(**overrides):
    """An engine like the configured one, with its own state (not the cached singleton)."""
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine import (
        EmbeddingWorkerEmbeddingEngine,
    )

    config = get_embedding_config()
    arguments = {
        "model": config.embedding_model,
        "dimensions": config.embedding_dimensions,
        "max_completion_tokens": config.embedding_max_completion_tokens,
    }
    arguments.update(overrides)
    return EmbeddingWorkerEmbeddingEngine(**arguments)


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

    texts = [*DOCUMENTS, "Where does Alice live?", "   "]
    result = await engine.embed_text(texts)
    dimensions = engine.get_vector_size()
    assert engine.worker_info is not None  # fetched with the readiness check
    assert dimensions == engine.worker_info.dimensions
    assert len(result) == len(texts)
    assert all(len(vector) == dimensions for vector in result)
    assert result[-1] == [0.0] * dimensions  # blank input
    assert all(abs(norm(vector) - 1.0) < 1e-3 for vector in result[:-1])
    assert result.tokens > 0
    query = result[3]
    assert cosine(query, result[0]) > max(cosine(query, result[1]), cosine(query, result[2]))
    assert engine.served_model == engine.worker_info.model

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
async def test_info_sets_the_input_limit_and_batch_size():
    from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (
        close_embedding_worker_transports,
    )

    engine = fresh_engine(batch_size=256)
    limit = await engine.input_limit()
    info = engine.worker_info
    assert info is not None and info.max_input_tokens > 0 and info.max_inputs > 0
    configured = engine.input_cap
    assert limit == (
        min(configured, info.max_input_tokens) if configured else info.max_input_tokens
    )
    assert engine.get_batch_size() == min(256, info.max_inputs)
    await close_embedding_worker_transports()


@needs_unset_dimensions
@pytest.mark.asyncio
async def test_dimensions_come_from_the_worker_when_unset():
    from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (
        close_embedding_worker_transports,
    )

    engine = fresh_engine(dimensions=None)
    assert engine.dimensions is None
    assert await engine.resolve_vector_size() == engine.worker_info.dimensions
    [vector] = await engine.embed_text(["hello"])
    assert len(vector) == engine.get_vector_size()
    await close_embedding_worker_transports()


@needs_unset_dimensions
def test_get_vector_size_before_any_contact_asks_the_worker():
    """The sync path cognee takes when it creates a collection first."""
    engine = fresh_engine(dimensions=None)
    assert engine.dimensions is None
    size = engine.get_vector_size()
    assert size == engine.worker_info.dimensions > 0


@pytest.mark.asyncio
async def test_configured_dimensions_that_differ_from_the_worker_are_fatal():
    from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (
        EmbeddingWorkerDimensionsMismatchError,
        close_embedding_worker_transports,
    )

    probe = fresh_engine(dimensions=None)
    served = await probe.resolve_vector_size() if probe.dimensions is None else probe.dimensions
    wrong = served * 2
    engine = fresh_engine(dimensions=wrong)
    for _ in range(2):  # latched
        with pytest.raises(EmbeddingWorkerDimensionsMismatchError) as raised:
            await engine.embed_text(["x"])
        assert str(served) in raised.value.message and str(wrong) in raised.value.message
    await close_embedding_worker_transports()


@pytest.mark.asyncio
async def test_an_over_long_input_is_split_and_averaged_into_a_unit_vector():
    from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (
        close_embedding_worker_transports,
    )

    engine = fresh_engine()
    await engine.input_limit()
    # Far past the worker's token limit: it answers input_too_long and the engine
    # splits the text, embeds the parts and pools them instead of failing.
    long_text = " ".join(DOCUMENTS * 20)
    [pooled] = await engine.embed_text([long_text])
    assert len(pooled) == engine.get_vector_size()
    assert abs(norm(pooled) - 1.0) < 1e-5
    # The pooled vector still means what the text says.
    [alice, rust, bananas] = list(await engine.embed_text(DOCUMENTS))
    assert min(cosine(pooled, alice), cosine(pooled, rust), cosine(pooled, bananas)) > 0.3
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
