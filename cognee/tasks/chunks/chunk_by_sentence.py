from uuid import uuid4, UUID
from typing import Optional, Iterator, Tuple
from .chunk_by_word import chunk_by_word


def chunk_by_sentence(
    data: str, maximum_length: Optional[int] = None
) -> Iterator[Tuple[UUID, str, int, Optional[str]]]:
    """
    Splits the input text into sentences based on word-level processing, with optional sentence length constraints.

    Notes:
        - Relies on the `chunk_by_word` function for word-level tokenization and classification.
        - Ensures sentences within paragraphs are uniquely identifiable using UUIDs.
        - Handles cases where the text ends mid-sentence by appending a special "sentence_cut" type.
    """
    sentence = ""
    paragraph_id = uuid4()
    word_count = 0
    section_end = False
    word_type_state = None

    # the yielded word_type_state is identical to word_type, except when
    # the word type is 'word', the word doesn't contain any letters
    # and words with the same characteristics connect it to a preceding
    # word with word_type 'paragraph_end' or 'sentence_end'
    for word, word_type in chunk_by_word(data):
        sentence += word
        word_count += 1

        if word_type in ["paragraph_end", "sentence_end"]:
            word_type_state = word_type
        else:
            for character in word:
                if character.isalpha():
                    word_type_state = word_type
                    break

        if word_type in ["paragraph_end", "sentence_end"] or (
            maximum_length and (word_count == maximum_length)
        ):
            yield (paragraph_id, sentence, word_count, word_type_state)
            sentence = ""
            word_count = 0
            paragraph_id = uuid4() if word_type == "paragraph_end" else paragraph_id

    if len(sentence) > 0:
        section_end = "sentence_cut" if word_type_state == "word" else word_type_state
        yield (
            paragraph_id,
            sentence,
            word_count,
            section_end,
        )
