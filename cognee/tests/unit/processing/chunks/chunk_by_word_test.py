import numpy as np
import pytest

from cognee.tasks.chunks import chunk_by_word
from cognee.tests.unit.processing.chunks.test_input import INPUT_TEXTS, INPUT_TEXTS_LONGWORDS


@pytest.mark.parametrize(
    "input_text",
    [
        INPUT_TEXTS["english_text"],
        INPUT_TEXTS["english_lists"],
        INPUT_TEXTS["python_code"],
        INPUT_TEXTS_LONGWORDS["chinese_text"],
    ],
)
def test_chunk_by_word_isomorphism(input_text):
    chunks = chunk_by_word(input_text)
    reconstructed_text = "".join([chunk[0] for chunk in chunks])
    assert reconstructed_text == input_text, (
        f"texts are not identical: {len(input_text) = }, {len(reconstructed_text) = }"
    )


@pytest.mark.parametrize(
    "input_text",
    [
        INPUT_TEXTS["english_text"],
        INPUT_TEXTS["english_lists"],
        INPUT_TEXTS["python_code"],
        INPUT_TEXTS_LONGWORDS["chinese_text"],
    ],
)
def test_chunk_by_word_splits(input_text):
    chunks = np.array(list(chunk_by_word(input_text)))
    space_test = np.array([" " not in chunk[0].strip() for chunk in chunks])

    assert np.all(space_test), f"These chunks contain spaces within them: {chunks[~space_test]}"
