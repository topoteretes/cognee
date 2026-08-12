"""Turn question texts into a normalized embedding matrix.

Phase 1 step 4 of the recall-coverage pipeline. Three jobs, in the order the
pipeline needs them:

* **Batch the calls.** ``EmbeddingEngine.embed_text`` is not internally batched —
  ``LiteLLMEmbeddingEngine`` puts the whole list into a single ``aembedding``
  request under a 30s timeout — so a 150-question run shipped in one call is one
  request that either works or loses everything. Mirrors
  ``cognee/modules/truth_subspace/build.py::_embed_in_batches``, including
  failing open per batch, except that a failed batch is padded with zero vectors
  of the engine's width rather than with ``[]``: an empty row makes the matrix
  ragged, and ``numpy`` raises on that instead of quietly dropping it.
* **Normalize.** Every row is L2-normalized so a dot product *is* the cosine
  similarity — the shape in
  ``cognee/tasks/memify/consolidate_entities.py::_normalize_rows``. Dedup,
  topic assignment and sink clustering then all reduce to one matmul.
* **Refuse degenerate embeddings.** ``MOCK_EMBEDDING=true`` makes every engine
  return all-zero vectors. Every pairwise similarity would be 0, dedup would
  find nothing, every question would look unique with ``occurrence_count = 1``,
  and the run would report a confident, fabricated coverage number. Raising is
  the only honest outcome, so :func:`embed_normalized` does.

Index alignment is the contract every caller depends on: row *i* of the returned
matrix is the vector for ``texts[i]``. A shifted row does not fail, it silently
attributes one question's meaning to another.
"""

from typing import Any, Optional, Sequence

import numpy as np

from cognee.modules.recall_coverage.exceptions import DegenerateEmbeddingError
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")


def unique_text_plan(texts: Sequence[str]) -> tuple[list[str], list[int]]:
    """Split ``texts`` into its distinct values plus a per-input index into them.

    Identical strings embed to identical vectors, so the same string is only ever
    sent once. This is also what makes the fan-out rule in
    :func:`cognee.modules.recall_coverage.dedup.collapse_asks` cost nothing: one
    search fanned across three datasets is three asks over one string, and the
    string is embedded once here rather than three times.

    The key is the exact text, not a normalized form, because the vector is a
    function of the exact bytes sent to the provider.
    """
    unique: list[str] = []
    index_of: dict[str, int] = {}
    plan: list[int] = []

    for text in texts:
        position = index_of.get(text)
        if position is None:
            position = len(unique)
            index_of[text] = position
            unique.append(text)
        plan.append(position)

    return unique, plan


def _vector_width(engine: Any) -> int:
    """The engine's advertised vector width, or 0 when it will not say.

    Only used to size the padding of a failed batch, so a missing or broken
    ``get_vector_size`` degrades to "infer the width from the batches that did
    succeed" rather than failing the run.
    """
    try:
        width = int(engine.get_vector_size())
    except Exception as error:  # pragma: no cover - engine-specific
        logger.debug("recall_coverage: engine did not report a vector size: %s", error)
        return 0
    return width if width > 0 else 0


def _resolve_batch_size(engine: Any, batch_size: Optional[int], text_count: int) -> int:
    """Batch size for the embedding calls: the argument, else the engine's own.

    Falls back to one request for everything when the engine will not name a
    batch size — that is exactly what an unbatched caller does today, so it
    cannot be a regression, and inventing a number here would be a magic number
    in a module whose thresholds are all config parameters.
    """
    if batch_size is not None and batch_size > 0:
        return int(batch_size)

    try:
        engine_batch_size = int(engine.get_batch_size())
    except Exception as error:  # pragma: no cover - engine-specific
        logger.debug("recall_coverage: engine did not report a batch size: %s", error)
        engine_batch_size = 0

    return engine_batch_size if engine_batch_size > 0 else max(text_count, 1)


def _rectangular(vectors: Sequence[Sequence[float]], width: int = 0) -> list[list[float]]:
    """Pad/truncate every row to one common width so ``numpy`` can take it.

    ``np.asarray`` raises on ragged input, and a run must not die because one
    provider response came back a row short. The common width is the widest row
    seen, or the engine's advertised width when that is larger.
    """
    observed = max((len(vector) for vector in vectors), default=0)
    common = max(observed, width)

    rows: list[list[float]] = []
    for vector in vectors:
        row = [float(value) for value in vector]
        if len(row) < common:
            row.extend([0.0] * (common - len(row)))
        elif len(row) > common:  # pragma: no cover - defensive
            row = row[:common]
        rows.append(row)
    return rows


async def embed_texts(
    engine: Any, texts: Sequence[str], *, batch_size: Optional[int] = None
) -> list[list[float]]:
    """Embed ``texts`` in bounded batches, index-aligned, rectangular.

    Fails open per batch: a batch that raises contributes zero vectors so that
    row *i* still belongs to ``texts[i]``. Those rows are cosine-similar to
    nothing, so the affected questions simply stop merging with anything instead
    of merging with the wrong thing.
    """
    if not texts:
        return []

    width = _vector_width(engine)
    size = _resolve_batch_size(engine, batch_size, len(texts))

    vectors: list[list[float]] = []
    for start in range(0, len(texts), size):
        batch = list(texts[start : start + size])
        try:
            batch_vectors = list(await engine.embed_text(batch))
        except Exception as error:
            logger.warning("recall_coverage: embedding batch failed open: %s", error)
            batch_vectors = []

        if len(batch_vectors) != len(batch):
            # An engine that returns a different number of vectors than it was
            # given has destroyed the alignment the caller relies on; pad or
            # trim back to the batch length instead of shifting every later row.
            logger.warning(
                "recall_coverage: embedding batch returned %s vectors for %s texts",
                len(batch_vectors),
                len(batch),
            )
            batch_vectors = batch_vectors[: len(batch)]
            batch_vectors.extend([[0.0] * width for _ in range(len(batch) - len(batch_vectors))])

        vectors.extend(list(vector) for vector in batch_vectors)

    return _rectangular(vectors, width)


def normalize_rows(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """L2-normalize each row so that a dot product equals cosine similarity.

    Returns a 2-D float array, or ``zeros((0, 0))`` when there is nothing to
    normalize. Zero rows are left as zeros: cosine-similar to nothing, which is
    what a failed embedding should be.
    """
    if not len(vectors):
        return np.zeros((0, 0))

    matrix = np.asarray(_rectangular(vectors), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:  # pragma: no cover - defensive
        return np.zeros((0, 0))

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def require_usable_embeddings(normalized: np.ndarray) -> np.ndarray:
    """Raise when every row of ``normalized`` is zero.

    See the module docstring: all-zero embeddings do not degrade dedup, they
    disable it while leaving the report looking normal.
    """
    if normalized.shape[0] == 0:
        return normalized

    if not np.any(np.linalg.norm(normalized, axis=1) > 0):
        raise DegenerateEmbeddingError()

    return normalized


async def embed_normalized(
    engine: Any, texts: Sequence[str], *, batch_size: Optional[int] = None
) -> np.ndarray:
    """Embed ``texts`` and return the L2-normalized, index-aligned matrix.

    Distinct strings are embedded once and the result is expanded back to one row
    per input, so row *i* is the vector for ``texts[i]`` even when several inputs
    share a string. Raises :class:`DegenerateEmbeddingError` when nothing came
    back usable.
    """
    if not texts:
        return np.zeros((0, 0))

    unique, plan = unique_text_plan(texts)
    unique_normalized = require_usable_embeddings(
        normalize_rows(await embed_texts(engine, unique, batch_size=batch_size))
    )

    if unique_normalized.shape[0] != len(unique):  # pragma: no cover - defensive
        raise DegenerateEmbeddingError(
            message="Embedding returned a different number of vectors than texts."
        )

    return unique_normalized[np.asarray(plan, dtype=int)]
