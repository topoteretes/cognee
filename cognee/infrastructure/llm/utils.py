import litellm

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    get_llm_client,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def get_max_chunk_tokens():
    """
    Calculate the maximum number of tokens allowed in a chunk.

    The function determines the maximum chunk size based on the maximum token limit of the
    embedding engine and half of the LLM maximum context token size. It ensures that the
    chunk size does not exceed these constraints.

    Returns:
    --------

        - int: The maximum number of tokens that can be included in a chunk, determined by
          the smaller value of the embedding engine's max tokens and half of the LLM's
          maximum tokens.
    """
    # NOTE: Import must be done in function to avoid circular import issue
    from cognee.infrastructure.databases.vector import get_vector_engine

    # Calculate max chunk size based on the following formula
    embedding_engine = get_vector_engine().embedding_engine
    llm_client = get_llm_client(raise_api_key_error=False)

    # We need to make sure chunk size won't take more than half of LLM max context token size
    # but it also can't be bigger than the embedding engine max token size
    llm_cutoff_point = llm_client.max_completion_tokens // 2  # Round down the division
    max_chunk_tokens = min(embedding_engine.max_completion_tokens, llm_cutoff_point)

    return max_chunk_tokens


def get_model_max_completion_tokens(model_name: str):
    """
    Retrieve the maximum token limit for a specified model name if it exists.

    Checks if the provided model name is present in the predefined model cost dictionary. If
    found, it logs the maximum token count for that model and returns it. If the model name
    is not recognized, it logs an informational message and returns None.

    Parameters:
    -----------

        - model_name (str): Name of LLM or embedding model

    Returns:
    --------

        Number of max tokens of model, or None if model is unknown
    """
    max_completion_tokens = None

    if model_name in litellm.model_cost:
        max_completion_tokens = litellm.model_cost[model_name]["max_tokens"]
        logger.debug(f"Max input tokens for {model_name}: {max_completion_tokens}")
    else:
        logger.debug("Model not found in LiteLLM's model_cost.")

    return max_completion_tokens


async def test_llm_connection():
    """
    Establish a connection to the LLM and create a structured output.

    Attempt to connect to the LLM client and uses the adapter to create a structured output
    with a predefined text input and system prompt. Log any exceptions encountered during
    the connection attempt and re-raise the exception for further handling.
    """
    try:
        await LLMGateway.acreate_structured_output(
            text_input="test",
            system_prompt='Respond to me with the following string: "test"',
            response_model=str,
        )

    except Exception as e:
        logger.error(e)
        logger.error("Connection to LLM could not be established.")
        raise e


async def test_embedding_connection():
    """
    Test the connection to the embedding engine by embedding a sample text.

    Handles exceptions that may occur during the operation, logs the error, and re-raises
    the exception if the connection to the embedding handler cannot be established.
    """
    try:
        # NOTE: Vector engine import must be done in function to avoid circular import issue
        from cognee.infrastructure.databases.vector import get_vector_engine

        await get_vector_engine().embedding_engine.embed_text("test")
    except Exception as e:
        logger.error(e)
        logger.error("Connection to Embedding handler could not be established.")
        raise e
