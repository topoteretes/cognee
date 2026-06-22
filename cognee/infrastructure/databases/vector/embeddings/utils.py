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


def sanitize_embedding_text_inputs(text: Union[str, List[str]]) -> List[str]:
    """
    Transform invalid/empty inputs into a safe dummy to prevent API 422 embedding errors while
    keeping list length consistent.
    """
    # Ensure we are working with a list
    text_list = [text] if isinstance(text, str) else text
    dummy_value = "."

    return [t if is_embeddable(t) else dummy_value for t in text_list]


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
