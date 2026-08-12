"""Guards on the recall-coverage embedding step.

Two invariants, both of which fail silently rather than loudly if broken:

* **Index alignment.** Row *i* of the matrix must be the vector for ``texts[i]``.
  A batch that fails, or an engine that returns the wrong number of vectors, must
  not shift every later row — that attributes one question's meaning to another
  and the run still reports a plausible number.
* **No all-zero embeddings.** ``MOCK_EMBEDDING=true`` returns all-zero vectors,
  under which every pairwise similarity is 0, dedup finds nothing and every
  question looks unique. This is the one place zero vectors belong: proving the
  guard raises.

The engines here are hand-written fakes with deterministic vectors rather than
``MOCK_EMBEDDING`` or ``MagicMock`` — the assertions are about exact vector
identity and exact call batching.
"""

import numpy as np
import pytest

from cognee.modules.recall_coverage.embedding import (
    embed_normalized,
    embed_texts,
    normalize_rows,
    require_usable_embeddings,
    unique_text_plan,
)
from cognee.modules.recall_coverage.exceptions import DegenerateEmbeddingError


class _FakeEngine:
    """Records the batches it was handed and returns one-hot-ish vectors."""

    def __init__(self, batch_size: int = 2, dimensions: int = 3, fail_batches=()):
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.fail_batches = set(fail_batches)
        self.batches: list[list[str]] = []

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_vector_size(self) -> int:
        return self.dimensions

    async def embed_text(self, texts):
        index = len(self.batches)
        self.batches.append(list(texts))
        if index in self.fail_batches:
            raise RuntimeError("provider exploded")
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class _ZeroEngine(_FakeEngine):
    """What ``MOCK_EMBEDDING=true`` behaves like."""

    async def embed_text(self, texts):
        self.batches.append(list(texts))
        return [[0.0] * self.dimensions for _ in texts]


class _ShortEngine(_FakeEngine):
    """A provider that returns fewer vectors than it was given."""

    async def embed_text(self, texts):
        self.batches.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in list(texts)[:-1]]


def test_unique_text_plan_embeds_each_string_once():
    unique, plan = unique_text_plan(["a", "b", "a", "c", "b"])

    assert unique == ["a", "b", "c"]
    assert plan == [0, 1, 0, 2, 1]


@pytest.mark.asyncio
async def test_embed_texts_chunks_by_the_engine_batch_size():
    engine = _FakeEngine(batch_size=2)

    vectors = await embed_texts(engine, ["aa", "bbb", "cccc", "d", "ee"])

    assert engine.batches == [["aa", "bbb"], ["cccc", "d"], ["ee"]]
    assert [vector[0] for vector in vectors] == [2.0, 3.0, 4.0, 1.0, 2.0]


@pytest.mark.asyncio
async def test_embed_texts_explicit_batch_size_wins():
    engine = _FakeEngine(batch_size=2)

    await embed_texts(engine, ["a", "b", "c"], batch_size=3)

    assert engine.batches == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_embed_texts_fails_open_per_batch_without_losing_alignment():
    engine = _FakeEngine(batch_size=2, fail_batches=(1,))

    vectors = await embed_texts(engine, ["aa", "bbb", "cccc", "dddddd", "ee"])

    assert len(vectors) == 5
    assert {len(vector) for vector in vectors} == {3}
    # The failed batch is zero-padded in place, so "ee" keeps its own vector.
    assert vectors[2] == [0.0, 0.0, 0.0]
    assert vectors[3] == [0.0, 0.0, 0.0]
    assert vectors[4][0] == 2.0


@pytest.mark.asyncio
async def test_embed_texts_pads_a_short_provider_response():
    engine = _ShortEngine(batch_size=3)

    vectors = await embed_texts(engine, ["a", "b", "c"])

    assert len(vectors) == 3
    assert vectors[-1] == [0.0, 0.0, 0.0]


def test_normalize_rows_makes_dot_product_the_cosine():
    normalized = normalize_rows([[3.0, 0.0], [0.0, 5.0], [1.0, 1.0]])

    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert normalized[0] @ normalized[1] == pytest.approx(0.0)
    assert normalized[0] @ normalized[2] == pytest.approx(1.0 / np.sqrt(2))


def test_normalize_rows_leaves_zero_rows_at_zero():
    normalized = normalize_rows([[0.0, 0.0], [1.0, 0.0]])

    assert list(normalized[0]) == [0.0, 0.0]
    assert normalized[0] @ normalized[1] == pytest.approx(0.0)


def test_normalize_rows_handles_an_empty_window():
    assert normalize_rows([]).shape == (0, 0)


def test_require_usable_embeddings_raises_when_every_norm_is_zero():
    with pytest.raises(DegenerateEmbeddingError):
        require_usable_embeddings(normalize_rows([[0.0, 0.0], [0.0, 0.0]]))


def test_require_usable_embeddings_allows_a_partial_failure():
    # One failed batch must not fail the run: those rows are cosine-similar to
    # nothing, so they stop merging rather than merging with the wrong thing.
    normalized = require_usable_embeddings(normalize_rows([[0.0, 0.0], [1.0, 0.0]]))

    assert normalized.shape == (2, 2)


@pytest.mark.asyncio
async def test_embed_normalized_is_index_aligned_and_reuses_repeated_text():
    engine = _FakeEngine(batch_size=10)

    normalized = await embed_normalized(engine, ["alpha", "beta", "alpha"])

    assert engine.batches == [["alpha", "beta"]]
    assert normalized.shape == (3, 3)
    assert np.allclose(normalized[0], normalized[2])
    assert not np.allclose(normalized[0], normalized[1])


@pytest.mark.asyncio
async def test_embed_normalized_raises_on_mock_style_zero_vectors():
    with pytest.raises(DegenerateEmbeddingError):
        await embed_normalized(_ZeroEngine(batch_size=4), ["alpha", "beta"])


@pytest.mark.asyncio
async def test_embed_normalized_handles_an_empty_window():
    engine = _FakeEngine()

    assert (await embed_normalized(engine, [])).shape == (0, 0)
    assert engine.batches == []
