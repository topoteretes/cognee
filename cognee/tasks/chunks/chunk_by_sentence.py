from collections.abc import Iterator
from uuid import UUID, uuid4

from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.shared.logging_utils import get_logger

from .chunk_by_word import chunk_by_word

logger = get_logger(__name__)


def get_word_size(word: str, tokenizer) -> int:
    """
    Calculate the size of a given word in terms of tokens.

    Counts the word's tokens with the embedding engine's tokenizer, or counts it as
    one token when the engine has no tokenizer.

    Parameters:
    -----------

        - word (str): The word for which the token size is to be calculated.
        - tokenizer: The embedding engine's tokenizer, or None.

    Returns:
    --------

        - int: The number of tokens representing the word.
    """
    if tokenizer:
        return tokenizer.count_tokens(word)
    return 1


def chunk_by_sentence(
    data: str, maximum_size: int | None = None
) -> Iterator[tuple[UUID, str, int, str | None]]:
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
    # The embedding engine is fixed for the whole call: it comes from the current
    # dataset's context, which cannot change mid-document. Resolving it per word
    # rebuilt the embedding config every time (a fastembed registry scan each word).
    tokenizer = get_embedding_engine().tokenizer
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
        word_size = get_word_size(word, tokenizer)

        if word_type in ["paragraph_end", "sentence_end"]:
            word_type_state = word_type
        else:
            for character in word:
                if character.isalpha():
                    word_type_state = word_type
                    break

        if maximum_size and (sentence_size + word_size > maximum_size):
            cut_type = "sentence_cut" if word_type_state == "word" else word_type_state
            yield (paragraph_id, sentence, sentence_size, cut_type)
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
        section_end = "sentence_cut" if word_type_state == "word" else word_type_state

        if maximum_size and sentence_size > maximum_size:
            # A trailing run with no sentence or paragraph boundary (a protein sequence, a
            # base64 blob, a glyph run left by PDF extraction) can exceed the chunk size. An
            # oversized run in the middle of a document is already emitted as a single chunk;
            # raising only when it is the last one made the whole document unprocessable, and
            # with no partial-commit mode that failed the entire pipeline run.
            logger.warning(
                "Trailing run of %d tokens exceeds the chunk size of %d; "
                "emitting it as a single oversized chunk.",
                sentence_size,
                maximum_size,
            )

        yield (
            paragraph_id,
            sentence,
            sentence_size,
            section_end,
        )
