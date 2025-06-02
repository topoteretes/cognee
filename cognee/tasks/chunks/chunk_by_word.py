import re
from typing import Iterator, Tuple


SENTENCE_ENDINGS = r"[.;!?…。！？]"
PARAGRAPH_ENDINGS = r"[\n\r]"


def is_real_paragraph_end(last_char: str, current_pos: int, text: str) -> bool:
    """
    Determine if the current position represents a valid paragraph end.

    The function checks if the last character indicates a possible sentence ending, then
    verifies if the subsequent characters lead to a valid paragraph end based on specific
    conditions.

    Parameters:
    -----------

        - last_char (str): The last processed character
        - current_pos (int): Current position in the text
        - text (str): The input text

    Returns:
    --------

        - bool: True if this is a real paragraph end, False otherwise
    """
    if re.match(SENTENCE_ENDINGS, last_char):
        return True
    j = current_pos + 1
    if j >= len(text):
        return False

    next_character = text[j]
    while j < len(text) and (re.match(PARAGRAPH_ENDINGS, next_character) or next_character == " "):
        j += 1
        if j >= len(text):
            return False
        next_character = text[j]

    if next_character.isupper():
        return True
    return False


def chunk_by_word(data: str) -> Iterator[Tuple[str, str]]:
    """
    Chunk text into words and sentence endings, preserving whitespace.

    Whitespace is included with the preceding word. Outputs can be joined with "" to
    recreate the original input.

    Parameters:
    -----------

        - data (str): The input string of text to be chunked into words and sentence
          endings.
    """
    current_chunk = ""
    i = 0

    while i < len(data):
        character = data[i]

        current_chunk += character

        if character == " ":
            yield (current_chunk, "word")
            current_chunk = ""
            i += 1
            continue

        if re.match(SENTENCE_ENDINGS, character):
            # Look ahead for whitespace
            next_i = i + 1
            while next_i < len(data) and data[next_i] == " ":
                current_chunk += data[next_i]
                next_i += 1

            is_paragraph_end = next_i < len(data) and re.match(PARAGRAPH_ENDINGS, data[next_i])
            yield (current_chunk, "paragraph_end" if is_paragraph_end else "sentence_end")
            current_chunk = ""
            i = next_i
            continue

        i += 1

    if current_chunk:
        yield (current_chunk, "word")
