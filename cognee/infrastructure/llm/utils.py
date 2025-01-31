import logging
import litellm

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client

logger = logging.getLogger(__name__)


def get_max_chunk_tokens():
    # Calculate max chunk size based on the following formula
    embedding_engine = get_vector_engine().embedding_engine
    llm_client = get_llm_client()

    # We need to make sure chunk size won't take more than half of LLM max context token size
    # but it also can't be bigger than the embedding engine max token size
    llm_cutoff_point = llm_client.max_tokens // 2  # Round down the division
    max_chunk_tokens = min(embedding_engine.max_tokens, llm_cutoff_point)

    return max_chunk_tokens


def get_model_max_tokens(model_name: str):
    """
    Args:
        model_name: name of LLM or embedding model

    Returns: Number of max tokens of model, or None if model is unknown
    """
    max_tokens = None

    if model_name in litellm.model_cost:
        max_tokens = litellm.model_cost[model_name]["max_tokens"]
        logger.debug(f"Max input tokens for {model_name}: {max_tokens}")
    else:
        logger.info("Model not found in LiteLLM's model_cost.")

    return max_tokens
