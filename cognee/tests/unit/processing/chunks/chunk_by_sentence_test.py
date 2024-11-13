import numpy as np
import pytest

from cognee.tasks.chunks import chunk_by_sentence, chunk_by_word
from cognee.tests.unit.processing.chunks.test_input import INPUT_TEXTS


@pytest.mark.parametrize(
    "input_text,maximum_length",
    [
        (INPUT_TEXTS["english_text"], None),
        (INPUT_TEXTS["english_text"], 8),
        (INPUT_TEXTS["english_text"], 64),
        (INPUT_TEXTS["english_lists"], None),
        (INPUT_TEXTS["english_lists"], 8),
        (INPUT_TEXTS["english_lists"], 64),
        (INPUT_TEXTS["python_code"], None),
        (INPUT_TEXTS["python_code"], 8),
        (INPUT_TEXTS["python_code"], 64),
        (INPUT_TEXTS["chinese_text"], None),
        (INPUT_TEXTS["chinese_text"], 8),
        (INPUT_TEXTS["chinese_text"], 64),
    ],
)
def test_chunk_by_sentence_isomorphism(input_text, maximum_length):
    chunks = chunk_by_sentence(input_text, maximum_length)
    reconstructed_text = "".join([chunk[2] for chunk in chunks])
    assert (
        reconstructed_text == input_text
    ), f"texts are not identical: {len(input_text) = }, {len(reconstructed_text) = }"


@pytest.mark.parametrize(
    "input_text,maximum_length",
    [
        (INPUT_TEXTS["english_text"], 8),
        (INPUT_TEXTS["english_text"], 64),
        (INPUT_TEXTS["english_lists"], 8),
        (INPUT_TEXTS["english_lists"], 64),
        (INPUT_TEXTS["python_code"], 8),
        (INPUT_TEXTS["python_code"], 64),
        (INPUT_TEXTS["chinese_text"], 8),
        (INPUT_TEXTS["chinese_text"], 64),
    ],
)
def test_paragraph_chunk_length(input_text, maximum_length):
    chunks = list(chunk_by_sentence(input_text, maximum_length))

    chunk_lengths = np.array([len(list(chunk_by_word(chunk[2]))) for chunk in chunks])

    larger_chunks = chunk_lengths[chunk_lengths > maximum_length]
    assert np.all(
        chunk_lengths <= maximum_length
    ), f"{maximum_length = }: {larger_chunks} are too large"
