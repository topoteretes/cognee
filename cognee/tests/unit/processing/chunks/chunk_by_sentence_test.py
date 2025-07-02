from itertools import product

import numpy as np
import pytest

from cognee.tasks.chunks import chunk_by_sentence
from cognee.tests.unit.processing.chunks.test_input import INPUT_TEXTS_LONGWORDS, INPUT_TEXTS
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

maximum_length_vals = [None, 16, 64]


@pytest.mark.parametrize(
    "input_text,maximum_length",
    list(product(list(INPUT_TEXTS.values()), maximum_length_vals)),
)
def test_chunk_by_sentence_isomorphism(input_text, maximum_length):
    chunks = chunk_by_sentence(input_text, maximum_length)
    reconstructed_text = "".join([chunk[1] for chunk in chunks])
    assert reconstructed_text == input_text, (
        f"texts are not identical: {len(input_text) = }, {len(reconstructed_text) = }"
    )


@pytest.mark.parametrize(
    "input_text,maximum_length",
    list(
        product(
            list(INPUT_TEXTS.values()),
            [val for val in maximum_length_vals if val is not None],
        )
    ),
)
def test_paragraph_chunk_length(input_text, maximum_length):
    chunks = list(chunk_by_sentence(input_text, maximum_length))

    embedding_engine = get_embedding_engine()
    chunk_lengths = np.array(
        [embedding_engine.tokenizer.count_tokens(chunk[1]) for chunk in chunks]
    )

    larger_chunks = chunk_lengths[chunk_lengths > maximum_length]
    assert np.all(chunk_lengths <= maximum_length), (
        f"{maximum_length = }: {larger_chunks} are too large"
    )


@pytest.mark.parametrize(
    "input_text,maximum_length",
    list(
        product(
            list(INPUT_TEXTS_LONGWORDS.values()),
            [val for val in maximum_length_vals if val is not None],
        )
    ),
)
def test_paragraph_chunk_long_input(input_text, maximum_length):
    with pytest.raises(ValueError):
        list(chunk_by_sentence(input_text, maximum_length))
