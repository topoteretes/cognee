from typing import List, Union
from cognee.shared.logging_utils import setup_logging

logger = setup_logging()


def is_embeddable(s: str) -> bool:
    """
    Check if input string is embeddable, if not it will be replaced with a dummy value to prevent API errors.
    Empty strings and a string with only a space character are not embeddable.
    If input string contains at least one alphanumeric character, it is considered embeddable.
    """
    if not isinstance(s, str):
        return False
    # Strip whitespace to check if the string is empty or only contains spaces
    s = s.strip()
    if len(s) >= 1:
        return True
    logger.debug(
        "Input string was not embeddable. Skipping embedding and using dummy value instead."
    )
    return False


def _strip_surrogates(s: str) -> str:
    """
    Replace unpaired UTF-16 surrogate code points that cannot be encoded to UTF-8.

    A lone/unpaired surrogate (e.g. a mis-decoded character from a Windows console or
    clipboard boundary) is a valid Python `str` but is not valid UTF-8. Left unstripped it
    crashes the tokenizer's `encode_batch()` with `TypeError: TextEncodeInput must be
    Union[...]` (and equivalent 422 errors in other embedding engines this function feeds),
    since embedding text is eventually encoded to bytes. Round-tripping through UTF-8 with
    `errors="replace"` removes/replaces surrogates while leaving normal text -- including
    valid multi-byte characters and properly paired surrogate emoji -- unchanged.
    """
    return s.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_embedding_text_inputs(text: Union[str, List[str]]) -> List[str]:
    """
    Transform invalid/empty inputs into a safe dummy to prevent API 422 embedding errors while
    keeping list length consistent.
    """
    # Ensure we are working with a list
    text_list = [text] if isinstance(text, str) else text
    dummy_value = "."

    return [_strip_surrogates(t) if is_embeddable(t) else dummy_value for t in text_list]


def handle_embedding_response(
    original_texts: Union[List[str], str], embeddings: List[List[float]], dimensions: int
) -> List[List[float]]:
    """
    Compare the original input strings against the results.
    If the original string was 'junk' that was not embeddable, overwrite its vector with zeros.
    """
    if isinstance(original_texts, str):
        original_texts = [original_texts]

    zero_vector = [0.0] * dimensions
    return [
        embeddings[i] if is_embeddable(original_texts[i]) else zero_vector
        for i in range(len(original_texts))
    ]
