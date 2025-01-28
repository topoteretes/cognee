from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client


def get_max_chunk_tokens():
    # Calculate max chunk size based on the following formula
    embedding_engine = get_vector_engine().embedding_engine
    llm_client = get_llm_client()

    # We need to make sure chunk size won't take more than half of LLM max context token size
    # but it also can't be bigger than the embedding engine max token size
    llm_cutoff_point = llm_client.max_tokens // 2  # Round down the division
    max_chunk_tokens = min(embedding_engine.max_tokens, llm_cutoff_point)

    return max_chunk_tokens
