from uuid import uuid4, UUID
from typing import Optional, Iterator, Tuple
from .chunk_by_word import chunk_by_word
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine


def get_word_size(word: str) -> int:
    embedding_engine = get_embedding_engine()
    if embedding_engine.tokenizer:
        return embedding_engine.tokenizer.count_tokens(word)
    else:
        return 1


def chunk_by_sentence(
    data: str, maximum_size: Optional[int] = None
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
    sentence_size = 0
    section_end = False
    word_type_state = None

    # the yielded word_type_state is identical to word_type, except when
    # the word type is 'word', the word doesn't contain any letters
    # and words with the same characteristics connect it to a preceding
    # word with word_type 'paragraph_end' or 'sentence_end'
    for word, word_type in chunk_by_word(data):
        word_size = get_word_size(word)

        if word_type in ["paragraph_end", "sentence_end"]:
            word_type_state = word_type
        else:
            for character in word:
                if character.isalpha():
                    word_type_state = word_type
                    break

        if maximum_size and (sentence_size + word_size > maximum_size):
            yield (paragraph_id, sentence, sentence_size, word_type_state)
            sentence = word
            sentence_size = word_size

        elif word_type in ["paragraph_end", "sentence_end"]:
            sentence += word
            sentence_size += word_size
            paragraph_id = uuid4() if word_type == "paragraph_end" else paragraph_id

            yield (paragraph_id, sentence, sentence_size, word_type_state)
            sentence = ""
            sentence_size = 0
        else:
            sentence += word
            sentence_size += word_size

    if len(sentence) > 0:
        if maximum_size and sentence_size > maximum_size:
            raise ValueError(f"Input word {word} longer than chunking size {maximum_size}.")

        section_end = "sentence_cut" if word_type_state == "word" else word_type_state
        yield (
            paragraph_id,
            sentence,
            sentence_size,
            section_end,
        )
