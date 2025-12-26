from itertools import product

import numpy as np
import pytest

from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.tasks.chunks import chunk_by_row

INPUT_TEXTS = "name: John, age: 30, city: New York, country: USA"
max_chunk_size_vals = [8, 32]


@pytest.mark.parametrize(
    "input_text,max_chunk_size",
    list(product([INPUT_TEXTS], max_chunk_size_vals)),
)
def test_chunk_by_row_isomorphism(input_text, max_chunk_size):
    chunks = chunk_by_row(input_text, max_chunk_size)
    reconstructed_text = ", ".join([chunk["text"] for chunk in chunks])
    assert reconstructed_text == input_text, (
        f"texts are not identical: {len(input_text) = }, {len(reconstructed_text) = }"
    )


@pytest.mark.parametrize(
    "input_text,max_chunk_size",
    list(product([INPUT_TEXTS], max_chunk_size_vals)),
)
def test_row_chunk_length(input_text, max_chunk_size):
    chunks = list(chunk_by_row(data=input_text, max_chunk_size=max_chunk_size))
    embedding_engine = get_embedding_engine()

    chunk_lengths = np.array(
        [embedding_engine.tokenizer.count_tokens(chunk["text"]) for chunk in chunks]
    )

    larger_chunks = chunk_lengths[chunk_lengths > max_chunk_size]
    assert np.all(chunk_lengths <= max_chunk_size), (
        f"{max_chunk_size = }: {larger_chunks} are too large"
    )


@pytest.mark.parametrize(
    "input_text,max_chunk_size",
    list(product([INPUT_TEXTS], max_chunk_size_vals)),
)
def test_chunk_by_row_chunk_numbering(input_text, max_chunk_size):
    chunks = chunk_by_row(data=input_text, max_chunk_size=max_chunk_size)
    chunk_indices = np.array([chunk["chunk_index"] for chunk in chunks])
    assert np.all(chunk_indices == np.arange(len(chunk_indices))), (
        f"{chunk_indices = } are not monotonically increasing"
    )
