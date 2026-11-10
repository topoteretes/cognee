from cognee.shared.logging_utils import setup_logging

logger = setup_logging()


def is_embeddable(s: str) -> bool:
    """
    Check if input string is embeddable, if not it will be replaced with a dummy value to prevent API errors.
    Empty strings and strings containing only whitespace are not embeddable.
    If input string contains at least one non-whitespace character, it is considered embeddable.
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


def sanitize_embedding_text_inputs(text: str | list[str]) -> list[str]:
    """
    Transform invalid/empty inputs into a safe dummy to prevent API 422 embedding errors while
    keeping list length consistent.
    """
    # Ensure we are working with a list
    text_list = [text] if isinstance(text, str) else text
    dummy_value = "."

    return [t if is_embeddable(t) else dummy_value for t in text_list]


def handle_embedding_response(
    original_texts: list[str] | str, embeddings: list[list[float]], dimensions: int
) -> list[list[float]]:
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


def count_input_tokens(response, texts: list[str], tokenizer) -> int:
    """Input tokens an embedding request consumed: the provider's own count when the
    response carries one, else the engine tokenizer's count of the texts sent.

    Best-effort: a counting failure is logged and counts 0, because raising here
    would fail (and retry, re-billing) an embedding request that succeeded.
    """
    try:
        reported = _reported_input_tokens(response)
        if reported is not None:
            return reported
        return sum(tokenizer.count_tokens(text) for text in texts)
    except Exception:
        logger.exception("Could not count embedding input tokens; counting 0")
        return 0


def _reported_input_tokens(response) -> int | None:
    """The provider's input-token count, or None when the response carries none:
    Ollama's prompt_eval_count, else prompt_tokens (else total_tokens) off usage.

    Accepts litellm's EmbeddingResponse and the openai SDK's CreateEmbeddingResponse
    (usage is an object) as well as plain dicts.
    """
    # Ollama's native /api/embed reports the count at the top level, not under usage.
    if isinstance(response, dict) and type(response.get("prompt_eval_count")) is int:
        return response["prompt_eval_count"]
    usage = (
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    )
    if usage is None:
        return None
    for field in ("prompt_tokens", "total_tokens"):
        value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None
