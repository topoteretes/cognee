import asyncio
import logging
import math
import os

import aiohttp
import aiohttp.http_exceptions
import litellm
import numpy as np
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_delay,
    wait_exponential_jitter,
)

from cognee.infrastructure.databases.exceptions import EmbeddingException
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.embeddings.input_limit import (
    init_input_limit,
    sane_limit,
)
from cognee.infrastructure.databases.vector.embeddings.retry_config import (
    embedding_retry_condition,
)
from cognee.infrastructure.databases.vector.embeddings.utils import (
    handle_embedding_response,
    sanitize_embedding_text_inputs,
)
from cognee.infrastructure.llm.exceptions import raise_if_budget_exhausted
from cognee.infrastructure.llm.tokenizer.resolver import resolve_embedding_tokenizer
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager
from cognee.shared.utils import create_secure_ssl_context

logger = get_logger("OllamaEmbeddingEngine")

observe = get_observe()


class OllamaEmbeddingEngine(EmbeddingEngine):
    """
    Implements an embedding engine using the Ollama embedding model.

    Public methods:
    - embed_text
    - get_vector_size
    - get_tokenizer

    Instance variables:
    - model
    - dimensions
    - max_completion_tokens
    - endpoint
    - mock
    - huggingface_tokenizer_name
    - tokenizer
    """

    model: str
    dimensions: int
    max_completion_tokens: int
    endpoint: str
    mock: bool
    huggingface_tokenizer_name: str

    def __init__(
        self,
        model: str | None = "avr/sfr-embedding-mistral:latest",
        dimensions: int | None = 1024,
        max_completion_tokens: int | None = None,
        endpoint: str | None = "http://localhost:11434/api/embed",
        huggingface_tokenizer: str = "Salesforce/SFR-Embedding-Mistral",
        batch_size: int = 100,
    ):
        self.model = model
        self.dimensions = dimensions
        self.endpoint = endpoint
        self.huggingface_tokenizer_name = huggingface_tokenizer
        self.batch_size = batch_size
        self.tokenizer = self.get_tokenizer()
        # Ollama embeds up to the model's context length and, by default, cuts
        # the rest without an error. The limit comes from /api/show (see
        # input_limit()), and every embed request below sends truncate=false so
        # an over-length input is rejected (and then embedded in parts) instead
        # of silently shortened.
        init_input_limit(self, max_completion_tokens)

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    @observe(as_type="embeddings")
    async def embed_text(self, text: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a list of text prompts.

        If mocking is enabled, returns a list of zero vectors instead of actual embeddings.

        Parameters:
        -----------

            - text (List[str]): A list of text prompts for which to generate embeddings.

        Returns:
        --------

            - List[List[float]]: A list of embedding vectors corresponding to the text prompts.
        """
        original_texts = text if isinstance(text, list) else [text]
        sanitized_text = sanitize_embedding_text_inputs(original_texts)

        if self.mock:
            embeddings = [[0.0] * self.dimensions for _ in sanitized_text]
            return handle_embedding_response(original_texts, embeddings, self.dimensions)

        try:
            embeddings = await asyncio.gather(
                *[self._get_embedding(prompt) for prompt in sanitized_text]
            )
            return handle_embedding_response(original_texts, embeddings, self.dimensions)
        except Exception as error:
            # A spend cap is terminal and is neither a context-window nor a
            # connectivity problem, so it must not fall through to the branches
            # below. Same actionable 402 the other engines raise.
            raise_if_budget_exhausted(error)

            error_str = str(error).lower()
            context_error_patterns = (
                "context length",
                "context window",
                "input length",
                "too long",
                "maximum context",
                "maximum tokens",
                "max tokens",
            )
            if any(pattern in error_str for pattern in context_error_patterns):
                if len(original_texts) > 1:
                    mid = math.ceil(len(original_texts) / 2)
                    left_vecs, right_vecs = await asyncio.gather(
                        self.embed_text(original_texts[:mid]),
                        self.embed_text(original_texts[mid:]),
                    )
                    embeddings = left_vecs + right_vecs
                    return handle_embedding_response(original_texts, embeddings, self.dimensions)

                if len(original_texts) == 1:
                    s = original_texts[0]
                    third = len(s) // 3
                    if third == 0:
                        raise EmbeddingException(
                            "Text is too short to split further but exceeds context window."
                        ) from error
                    left_part, right_part = s[: third * 2], s[third:]
                    (left_vec,), (right_vec,) = await asyncio.gather(
                        self.embed_text([left_part]),
                        self.embed_text([right_part]),
                    )
                    pooled = (np.array(left_vec) + np.array(right_vec)) / 2
                    embeddings = [pooled.tolist()]
                    return handle_embedding_response(original_texts, embeddings, self.dimensions)

                return handle_embedding_response(original_texts, embeddings, self.dimensions)

            logger.error(f"Embedding error in OllamaEmbeddingEngine: {error!s}")
            raise EmbeddingException(
                f"Failed to index data points using model {self.model}"
            ) from error

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(8, 128),
        # Budget exhaustion is terminal here too: EMBEDDING_ENDPOINT is not
        # validated for provider shape, and a proxy's OpenAI-shaped response
        # body is handled by the "data" branch below, so this engine can be
        # pointed at a LiteLLM proxy and reach its spend cap. The rejection
        # arrives as a RuntimeError built from the error body, which no type
        # tuple can match -- see embeddings/retry_config.py.
        #
        # Not at parity with the other two engines: rebuilding the failure as a
        # bare RuntimeError drops the status code and the response object, so
        # only the message-text signal survives. A rejection whose body carries
        # ``type: budget_exceeded`` but no budget sentence, or a reworded or
        # truncated sentence, is not classified here and runs the full ladder.
        # Closing that means classifying inside ``_get_embedding``, while the
        # status and the parsed body are still in hand.
        retry=embedding_retry_condition(
            litellm.exceptions.NotFoundError, ValueError, asyncio.CancelledError
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get_embedding(self, prompt: str) -> list[float]:
        """
        Internal method to call the Ollama embeddings endpoint for a single prompt.
        """

        payload = {
            "model": self.model,
            "input": prompt,
            "dimensions": self.dimensions,
        }
        if "/api/" in self.endpoint:
            # Ollama's native API: reject over-length input ("the input length
            # exceeds the context length") instead of embedding only the head.
            # The rejection is classified below and embed_text splits the text.
            # Not sent to an OpenAI-shaped endpoint (a LiteLLM proxy, /v1/...),
            # which rejects unknown fields.
            payload["truncate"] = False

        headers = {}
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with (
            aiohttp.ClientSession(connector=connector) as session,
            embedding_rate_limiter_context_manager(),
            session.post(self.endpoint, json=payload, headers=headers, timeout=60.0) as response,
        ):
            data = await response.json()

            if "error" in data:
                # The body shape varies by server: Ollama sends a string,
                # an OpenAI-compatible proxy sends an object, and either
                # can send null. Coerce before the membership tests below,
                # which would otherwise test dict keys (always False) or
                # raise TypeError on None, turning a terminal over-length
                # error into a retryable one that burns the full ladder.
                error_msg = str(data["error"])
                if "context length" in error_msg or "input length" in error_msg:
                    # Expected with truncate=false: embed_text embeds the text in
                    # parts, so this is recovery, not a failure.
                    logger.debug(
                        "Ollama rejected an over-length input for %s (%s); embedding it in parts.",
                        self.model,
                        error_msg,
                    )
                    raise ValueError(f"Text too long for embedding model: {error_msg}")
                logger.error(f"Ollama embedding error: {error_msg}")
                raise RuntimeError(f"Ollama embedding API error: {error_msg}")

            if "embeddings" in data:
                return data["embeddings"][0]
            elif "embedding" in data:
                return data["embedding"]
            elif "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            else:
                raise ValueError(f"Unexpected response format from Ollama: {data}")

    def get_vector_size(self) -> int:
        """
        Retrieve the size of the embedding vectors.

        Returns:
        --------

            - int: The dimension of the embedding vectors.
        """
        return self.dimensions

    def get_batch_size(self) -> int:
        """
        Return the desired batch size for embedding calls

        Returns:

        """
        return self.batch_size

    input_limit_source = "Ollama /api/show context_length"

    async def input_limit(self) -> int | None:
        """The model's context length from Ollama's ``/api/show``, or None.

        Ollama embeds up to the model's ``context_length``. Asked asynchronously so
        a slow server holds up only this request, not the whole event loop; httpx's
        default timeout bounds the wait. Anything that stops the lookup, from a
        server that is down to an endpoint that is not Ollama-shaped, gives None:
        the cap then stands, and ``truncate: false`` on every embed request still
        turns an over-length input into an error instead of a silent cut.
        """
        if not self.endpoint or not self.model or "/api/" not in self.endpoint:
            return None
        import httpx

        show_url = f"{self.endpoint.split('/api/', 1)[0]}/api/show"
        api_key = os.getenv("LLM_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(show_url, json={"model": self.model}, headers=headers)
            response.raise_for_status()
            model_info = response.json().get("model_info") or {}
        except Exception:
            logger.debug(
                "Could not read the context length of %r from %s",
                self.model,
                show_url,
                exc_info=True,
            )
            return None
        for key, value in model_info.items():
            if key.endswith(".context_length"):
                return sane_limit(value)
        return None

    def get_tokenizer(self):
        """
        Load and return the tokenizer for the embedding engine.

        An Ollama model id is not a HuggingFace repo, so the configured
        HUGGINGFACE_TOKENIZER override selects the tokenizer; resolution warns on
        mismatch and falls back safely to TikToken (issue #3646).

        Returns:
        --------

            The tokenizer used by the embedding engine.
        """
        logger.debug("Loading tokenizer for OllamaEmbeddingEngine...")
        tokenizer = resolve_embedding_tokenizer(
            provider="ollama",
            model=self.model,
            huggingface_tokenizer=self.huggingface_tokenizer_name,
        )
        logger.debug("Tokenizer loaded for OllamaEmbeddingEngine")
        return tokenizer
