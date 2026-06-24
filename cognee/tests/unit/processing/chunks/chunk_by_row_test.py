from itertools import product

import numpy as np
import pytest

from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.tasks.chunks import chunk_by_row

INPUT_TEXTS = "name: John, age: 30, city: New York, country: USA"
max_chunk_size_vals = [8, 32]

# Two rows separated by a blank line. chunk_by_row splits its input on
# "\n\n", so a single call can legitimately receive several rows (e.g. when
# CsvChunker is applied to a multi-paragraph TextDocument, or chunk_by_row is
# called directly). Each row must become its own chunk.
MULTI_ROW_INPUT = "name: John, age: 30\n\nname: Jane, age: 25"


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


def test_chunk_by_row_multiple_rows_are_independent():
    """Each "\\n\\n"-separated row must produce an independent chunk.

    Regression test: row state (text/size) was not reset after yielding a
    "row_end" chunk, so subsequent rows accumulated previous rows' pairs and
    sizes, and the chunk index was never advanced. This corrupted the stored
    chunk text and broke the reconstruction invariant.
    """
    chunks = list(chunk_by_row(data=MULTI_ROW_INPUT, max_chunk_size=128))

    # One chunk per row, no leakage of one row's pairs into the next.
    assert [chunk["text"] for chunk in chunks] == [
        "name: John, age: 30",
        "name: Jane, age: 25",
    ]

    # Concatenating the chunks reproduces the original rows exactly.
    reconstructed = "\n\n".join(chunk["text"] for chunk in chunks)
    assert reconstructed == MULTI_ROW_INPUT

    # Indices are monotonically increasing across rows.
    chunk_indices = [chunk["chunk_index"] for chunk in chunks]
    assert chunk_indices == list(range(len(chunk_indices)))

    # Identical rows must have identical sizes (no accumulation across rows).
    assert chunks[0]["chunk_size"] == chunks[1]["chunk_size"]
