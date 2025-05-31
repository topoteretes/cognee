from uuid import uuid4, UUID
from typing import Optional, Iterator, Tuple
from .chunk_by_word import chunk_by_word
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine


def get_word_size(word: str) -> int:
    """
    Calculate the size of a given word in terms of tokens.

    If an embedding engine's tokenizer is available, count the tokens for the provided word.
    If the tokenizer is not available, assume the word counts as one token.

    Parameters:
    -----------

        - word (str): The word for which the token size is to be calculated.

    Returns:
    --------

        - int: The number of tokens representing the word, typically an integer, depending
          on the tokenizer's output.
    """
    embedding_engine = get_embedding_engine()
    if embedding_engine.tokenizer:
        return embedding_engine.tokenizer.count_tokens(word)
    else:
        return 1


def chunk_by_sentence(
    data: str, maximum_size: Optional[int] = None
) -> Iterator[Tuple[UUID, str, int, Optional[str]]]:
    """
    Splits text into sentences while preserving word and paragraph boundaries.

    This function processes the input string, dividing it into sentences based on word-level
    tokenization. Each sentence is identified with a unique UUID, and it handles scenarios
    where the text may end mid-sentence by tagging it with a specific type. If a maximum
    sentence length is specified, the function ensures that sentences do not exceed this
    length, raising a ValueError if an individual word surpasses it. The function utilizes
    an external word processing function `chunk_by_word` to determine the structure of the
    text.

    Parameters:
    -----------

        - data (str): The input text to be split into sentences.
        - maximum_size (Optional[int]): An optional limit on the maximum size of sentences
          generated. (default None)
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
