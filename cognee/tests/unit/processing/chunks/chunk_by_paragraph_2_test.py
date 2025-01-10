from itertools import product

import numpy as np
import pytest

from cognee.tasks.chunks import chunk_by_paragraph, chunk_by_word
from cognee.tests.unit.processing.chunks.test_input import INPUT_TEXTS

paragraph_lengths = [64, 256, 1024]
batch_paragraphs_vals = [True, False]


@pytest.mark.parametrize(
    "input_text,paragraph_length,batch_paragraphs",
    list(product(list(INPUT_TEXTS.values()), paragraph_lengths, batch_paragraphs_vals)),
)
def test_chunk_by_paragraph_isomorphism(input_text, paragraph_length, batch_paragraphs):
    chunks = chunk_by_paragraph(input_text, paragraph_length, batch_paragraphs)
    reconstructed_text = "".join([chunk["text"] for chunk in chunks])
    assert reconstructed_text == input_text, (
        f"texts are not identical: {len(input_text) = }, {len(reconstructed_text) = }"
    )


@pytest.mark.parametrize(
    "input_text,paragraph_length,batch_paragraphs",
    list(product(list(INPUT_TEXTS.values()), paragraph_lengths, batch_paragraphs_vals)),
)
def test_paragraph_chunk_length(input_text, paragraph_length, batch_paragraphs):
    chunks = list(
        chunk_by_paragraph(
            data=input_text, paragraph_length=paragraph_length, batch_paragraphs=batch_paragraphs
        )
    )

    chunk_lengths = np.array([len(list(chunk_by_word(chunk["text"]))) for chunk in chunks])

    larger_chunks = chunk_lengths[chunk_lengths > paragraph_length]
    assert np.all(chunk_lengths <= paragraph_length), (
        f"{paragraph_length = }: {larger_chunks} are too large"
    )


@pytest.mark.parametrize(
    "input_text,paragraph_length,batch_paragraphs",
    list(product(list(INPUT_TEXTS.values()), paragraph_lengths, batch_paragraphs_vals)),
)
def test_chunk_by_paragraph_chunk_numbering(input_text, paragraph_length, batch_paragraphs):
    chunks = chunk_by_paragraph(
        data=input_text, paragraph_length=paragraph_length, batch_paragraphs=batch_paragraphs
    )
    chunk_indices = np.array([chunk["chunk_index"] for chunk in chunks])
    assert np.all(chunk_indices == np.arange(len(chunk_indices))), (
        f"{chunk_indices = } are not monotonically increasing"
    )
