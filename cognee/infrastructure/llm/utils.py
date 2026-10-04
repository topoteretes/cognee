import asyncio
import os

import litellm

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger

logger = get_logger()

CONNECTION_TEST_TIMEOUT_SECONDS = 30


async def get_max_chunk_tokens() -> int:
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
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.embeddings.input_limit import (
        resolve_input_limit,
    )
    from cognee.infrastructure.llm.config import get_llm_context_config

    # Calculate max chunk size based on the following formula
    embedding_engine = (await get_vector_engine_async()).embedding_engine
    await resolve_input_limit(embedding_engine)

    # Resolve the LLM token ceiling from configuration alone — building an LLM
    # client here would eagerly instantiate the legacy framework's adapter
    # even when the litellm_native framework is active. Mirrors the resolution
    # in get_llm_client()/get_native_client(): the lower of the model's hard
    # limit and the user's configured ceiling, with the Ollama context-size
    # special case.
    llm_config = get_llm_context_config()
    model_max = get_model_max_completion_tokens(llm_config.llm_model)
    if model_max is not None:
        llm_max_completion_tokens = min(model_max, llm_config.llm_max_completion_tokens)
    elif llm_config.llm_provider == "ollama":
        llm_max_completion_tokens = llm_config.ollama_num_ctx
    else:
        llm_max_completion_tokens = llm_config.llm_max_completion_tokens

    # We need to make sure chunk size won't take more than half of LLM max context token size
    # but it also can't be bigger than the embedding engine max token size
    llm_cutoff_point = llm_max_completion_tokens // 2  # Round down the division
    max_chunk_tokens = min(embedding_engine.max_completion_tokens, llm_cutoff_point)

    return max_chunk_tokens


# (chunk_size, limit) pairs already warned about: a pipeline resolves the size in
# more than one place, and one warning per process says everything the next would.
_chunk_size_warnings_issued: set[tuple[int, int]] = set()


async def resolve_chunk_size(chunk_size: int | None) -> int:
    """The chunk size a pipeline runs with: the caller's, capped at what can be embedded.

    ``None`` (or 0) means the automatic size from :func:`get_max_chunk_tokens`. An
    explicit value is kept unless it exceeds the embedding engine's token limit --
    the model's own input limit, or ``EMBEDDING_MAX_COMPLETION_TOKENS`` when that
    is lower -- in which case it is lowered to that limit with a warning: text past
    the limit would be dropped from the embedding, or rejected by the provider.
    The LLM half-context rule only applies to the automatic size.
    """
    if not chunk_size:
        return await get_max_chunk_tokens()
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be a positive number of tokens, got {chunk_size}")

    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.embeddings.input_limit import (
        resolve_input_limit,
    )

    embedding_engine = (await get_vector_engine_async()).embedding_engine
    limit = await resolve_input_limit(embedding_engine)
    if chunk_size <= limit:
        return chunk_size

    model_limit = embedding_engine.model_input_limit
    if (chunk_size, limit) in _chunk_size_warnings_issued:
        return limit
    _chunk_size_warnings_issued.add((chunk_size, limit))
    if limit == model_limit:
        logger.warning(
            "chunk_size=%s exceeds what embedding model %r accepts (%s tokens of text); using %s. "
            "Text beyond the model's limit would be dropped from the embedding.",
            chunk_size,
            embedding_engine.model,
            model_limit,
            limit,
        )
    else:
        logger.warning(
            "chunk_size=%s exceeds EMBEDDING_MAX_COMPLETION_TOKENS (%s); using %s. Raise "
            "EMBEDDING_MAX_COMPLETION_TOKENS to allow larger chunks%s.",
            chunk_size,
            limit,
            limit,
            f" (embedding model {embedding_engine.model!r} accepts up to {model_limit} tokens of text)"
            if model_limit is not None
            else "",
        )
    return limit


def get_model_max_completion_tokens(model_name: str) -> int | None:
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
    max_completion_tokens: int | None = None

    if model_name in litellm.model_cost:
        if "max_tokens" in litellm.model_cost[model_name]:
            max_completion_tokens = litellm.model_cost[model_name]["max_tokens"]
            logger.debug(f"Max input tokens for {model_name}: {max_completion_tokens}")
        else:
            logger.debug(
                f"Model max_tokens not found in LiteLLM's model_cost for model {model_name}."
            )
    else:
        logger.debug("Model not found in LiteLLM's model_cost.")

    return max_completion_tokens


async def test_llm_connection() -> None:
    """
    Test connectivity to the LLM endpoint using a simple completion call.
    """
    try:
        logger.info("Testing connection to LLM endpoint...")
        await asyncio.wait_for(
            LLMGateway.acreate_structured_output(
                text_input="test",
                system_prompt='Respond to me with the following string: "test"',
                response_model=str,
            ),
            timeout=CONNECTION_TEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        msg = (
            f"LLM connection test timed out after {CONNECTION_TEST_TIMEOUT_SECONDS}s. "
            "Check that your LLM endpoint is reachable and responding. "
            "Set COGNEE_SKIP_CONNECTION_TEST=true to bypass this check."
        )
        logger.error(msg)
        raise TimeoutError(msg)
    except litellm.exceptions.AuthenticationError:
        msg = (
            "LLM authentication failed. Check your LLM_API_KEY configuration. "
            "Set COGNEE_SKIP_CONNECTION_TEST=true to bypass this check."
        )
        logger.error(msg)
        raise
    except Exception as e:
        logger.error(e)
        logger.error("Connection to LLM could not be established.")
        raise


async def test_embedding_connection() -> int:
    """
    Test the connection to the embedding engine by embedding a sample text.
    Returns detected embedding vector dimensions from the test response.

    Handles exceptions that may occur during the operation, logs the error, and re-raises
    the exception if the connection to the embedding handler cannot be established.
    Wrapped in a timeout to prevent indefinite hangs.
    """
    try:
        # NOTE: Vector engine import must be done in function to avoid circular import issue
        from cognee.infrastructure.databases.vector import get_vector_engine_async
        from cognee.infrastructure.databases.vector.embeddings.input_limit import (
            resolve_input_limit,
        )

        logger.info("Testing connection to Embedding endpoint...")
        vector_engine = await get_vector_engine_async()
        # Learn the model's input limit up front, so its log line appears at
        # startup and the first chunk-size resolution finds it ready.
        await resolve_input_limit(vector_engine.embedding_engine)
        embedding_vectors = await asyncio.wait_for(
            vector_engine.embedding_engine.embed_text(["test"]),
            timeout=CONNECTION_TEST_TIMEOUT_SECONDS,
        )

        if not embedding_vectors or not embedding_vectors[0]:
            raise ValueError("Embedding test did not return a valid vector.")

        return len(embedding_vectors[0])
    except asyncio.TimeoutError:
        msg = (
            f"Embedding connection test timed out after {CONNECTION_TEST_TIMEOUT_SECONDS}s. "
            "Check that your embedding endpoint is reachable. "
            "Set COGNEE_SKIP_CONNECTION_TEST=true to bypass this check."
        )
        logger.error(msg)
        raise TimeoutError(msg)
    except Exception as e:
        logger.error(e)
        logger.error("Connection to Embedding handler could not be established.")
        raise


async def determine_embedding_dimensions(detected_dimensions: int) -> None:
    """
    Apply embedding-dimension policy using a single already-produced test vector size.

    Rules:
    - If EMBEDDING_DIMENSIONS is explicitly provided by the user, keep it as source of truth.
    - Otherwise, sync config and active embedding engine dimensions to detected size.
    """
    configured_dimensions_raw = os.getenv("EMBEDDING_DIMENSIONS")
    if configured_dimensions_raw is not None and configured_dimensions_raw.strip():
        return

    # NOTE: Imports inside function to avoid circular imports.
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config

    embedding_config = get_embedding_config()
    if embedding_config.embedding_dimensions != detected_dimensions:
        embedding_config.embedding_dimensions = detected_dimensions

        # Keep active engine in sync in this process.
        embedding_engine = (await get_vector_engine_async()).embedding_engine
        if hasattr(embedding_engine, "dimensions"):
            embedding_engine.dimensions = detected_dimensions

        logger.info(
            "Auto-detected embedding dimensions from connection test: %s",
            detected_dimensions,
        )
