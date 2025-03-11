from itertools import product

import numpy as np
import pytest

from cognee.tests.unit.processing.chunks.test_input import INPUT_TEXTS
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.tasks.chunks import chunk_by_paragraph

batch_paragraphs_vals = [True, False]
max_chunk_size_vals = [512, 1024, 4096]


@pytest.mark.parametrize(
    "input_text,max_chunk_size,batch_paragraphs",
    list(
        product(
            list(INPUT_TEXTS.values()),
            max_chunk_size_vals,
            batch_paragraphs_vals,
        )
    ),
)
def test_chunk_by_paragraph_isomorphism(input_text, max_chunk_size, batch_paragraphs):
    chunks = chunk_by_paragraph(input_text, max_chunk_size, batch_paragraphs)
    reconstructed_text = "".join([chunk["text"] for chunk in chunks])
    assert reconstructed_text == input_text, (
        f"texts are not identical: {len(input_text) = }, {len(reconstructed_text) = }"
    )


@pytest.mark.parametrize(
    "input_text,max_chunk_size, batch_paragraphs",
    list(
        product(
            list(INPUT_TEXTS.values()),
            max_chunk_size_vals,
            batch_paragraphs_vals,
        )
    ),
)
def test_paragraph_chunk_length(input_text, max_chunk_size, batch_paragraphs):
    chunks = list(
        chunk_by_paragraph(
            data=input_text,
            max_chunk_size=max_chunk_size,
            batch_paragraphs=batch_paragraphs,
        )
    )
    embedding_engine = get_embedding_engine()

    chunk_lengths = np.array(
        [embedding_engine.tokenizer.count_tokens(chunk["text"]) for chunk in chunks]
    )

    larger_chunks = chunk_lengths[chunk_lengths > max_chunk_size]
    assert np.all(chunk_lengths <= max_chunk_size), (
        f"{max_chunk_size = }: {larger_chunks} are too large"
    )


@pytest.mark.parametrize(
    "input_text,max_chunk_size,batch_paragraphs",
    list(
        product(
            list(INPUT_TEXTS.values()),
            max_chunk_size_vals,
            batch_paragraphs_vals,
        )
    ),
)
def test_chunk_by_paragraph_chunk_numbering(input_text, max_chunk_size, batch_paragraphs):
    chunks = chunk_by_paragraph(
        data=input_text,
        max_chunk_size=max_chunk_size,
        batch_paragraphs=batch_paragraphs,
    )
    chunk_indices = np.array([chunk["chunk_index"] for chunk in chunks])
    assert np.all(chunk_indices == np.arange(len(chunk_indices))), (
        f"{chunk_indices = } are not monotonically increasing"
    )
