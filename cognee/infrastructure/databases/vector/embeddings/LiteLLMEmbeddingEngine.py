import asyncio
import logging

from cognee.shared.logging_utils import get_logger
from typing import List, Optional
import numpy as np
import math
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)
import litellm
import os
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions import EmbeddingException
from cognee.infrastructure.llm.tokenizer.HuggingFace import (
    HuggingFaceTokenizer,
)
from cognee.infrastructure.llm.tokenizer.Mistral import (
    MistralTokenizer,
)
from cognee.infrastructure.llm.tokenizer.TikToken import (
    TikTokenTokenizer,
)

litellm.set_verbose = False
logger = get_logger("LiteLLMEmbeddingEngine")


class LiteLLMEmbeddingEngine(EmbeddingEngine):
    """
    Engine for embedding text using a specific LLM model, supporting mock and actual
    embedding calls.

    Public methods:
    - embed_text: Embed a list of strings into vector representations.
    - get_vector_size: Retrieve the size of the embedding vectors.
    - get_tokenizer: Load the appropriate tokenizer for the specified model.
    """

    api_key: str
    endpoint: str
    api_version: str
    provider: str
    model: str
    dimensions: int
    mock: bool

    MAX_RETRIES = 5

    def __init__(
        self,
        model: Optional[str] = "openai/text-embedding-3-large",
        provider: str = "openai",
        dimensions: Optional[int] = 3072,
        api_key: str = None,
        endpoint: str = None,
        api_version: str = None,
        max_completion_tokens: int = 512,
        batch_size: int = 100,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.max_completion_tokens = max_completion_tokens
        self.tokenizer = self.get_tokenizer()
        self.retry_count = 0
        self.batch_size = batch_size

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings into vector representations.

        If the input exceeds the model's context window, the method will recursively split the
        input and combine the results. It handles both mock and live embedding scenarios,
        logging errors for any encountered exceptions, and raising specific exceptions for
        context window issues and embedding failures.

        Parameters:
        -----------

            - text (List[str]): A list of strings to be embedded.

        Returns:
        --------

            - List[List[float]]: A list of vectors representing the embedded texts.
        """
        try:
            if self.mock:
                response = {"data": [{"embedding": [0.0] * self.dimensions} for _ in text]}
                return [data["embedding"] for data in response["data"]]
            else:
                response = await litellm.aembedding(
                    model=self.model,
                    input=text,
                    api_key=self.api_key,
                    api_base=self.endpoint,
                    api_version=self.api_version,
                )

                return [data["embedding"] for data in response.data]

        except litellm.exceptions.ContextWindowExceededError as error:
            if isinstance(text, list) and len(text) > 1:
                mid = math.ceil(len(text) / 2)
                left, right = text[:mid], text[mid:]
                left_vecs, right_vecs = await asyncio.gather(
                    self.embed_text(left),
                    self.embed_text(right),
                )
                return left_vecs + right_vecs

                # If caller passed ONE oversize string split the string itself into
                # half so we can process it
            if isinstance(text, list) and len(text) == 1:
                logger.debug(f"Pooling embeddings of text string with size: {len(text[0])}")
                s = text[0]
                third = len(s) // 3
                # We are using thirds to intentionally have overlap between split parts
                # for better embedding calculation
                left_part, right_part = s[: third * 2], s[third:]

                # Recursively embed the split parts in parallel
                (left_vec,), (right_vec,) = await asyncio.gather(
                    self.embed_text([left_part]),
                    self.embed_text([right_part]),
                )

                # POOL the two embeddings into one
                pooled = (np.array(left_vec) + np.array(right_vec)) / 2
                return [pooled.tolist()]

            logger.error("Context window exceeded for embedding text: %s", str(error))
            raise error

        except (
            litellm.exceptions.BadRequestError,
            litellm.exceptions.NotFoundError,
        ) as e:
            logger.error(f"Embedding error with model {self.model}: {str(e)}")
            raise EmbeddingException(f"Failed to index data points using model {self.model}") from e

        except Exception as error:
            logger.error("Error embedding text: %s", str(error))
            raise error

    def get_vector_size(self) -> int:
        """
        Retrieve the dimensionality of the embedding vectors.

        Returns:
        --------

            - int: The size (dimensionality) of the embedding vectors.
        """
        return self.dimensions

    def get_batch_size(self) -> int:
        """
        Return the desired batch size for embedding calls

        Returns:

        """
        return self.batch_size

    def get_tokenizer(self):
        """
        Load and return the appropriate tokenizer for the specified model based on the provider.

        Returns:
        --------

            The tokenizer instance compatible with the model.
        """
        logger.debug(f"Loading tokenizer for model {self.model}...")
        # If model also contains provider information, extract only model information
        model = self.model.split("/")[-1]

        if "openai" in self.provider.lower():
            tokenizer = TikTokenTokenizer(
                model=model, max_completion_tokens=self.max_completion_tokens
            )
        elif "gemini" in self.provider.lower():
            # Since Gemini tokenization needs to send an API request to get the token count we will use TikToken to
            # count tokens as we calculate tokens word by word
            tokenizer = TikTokenTokenizer(
                model=None, max_completion_tokens=self.max_completion_tokens
            )
            # Note: Gemini Tokenizer expects an LLM model as input and not the embedding model
            # tokenizer = GeminiTokenizer(
            #     llm_model=llm_model, max_completion_tokens=self.max_completion_tokens
            # )
        elif "mistral" in self.provider.lower():
            tokenizer = MistralTokenizer(
                model=model, max_completion_tokens=self.max_completion_tokens
            )
        else:
            try:
                tokenizer = HuggingFaceTokenizer(
                    model=self.model.replace("hosted_vllm/", ""),
                    max_completion_tokens=self.max_completion_tokens,
                )
            except Exception as e:
                logger.warning(f"Could not get tokenizer from HuggingFace due to: {e}")
                logger.info("Switching to TikToken default tokenizer.")
                tokenizer = TikTokenTokenizer(
                    model=None, max_completion_tokens=self.max_completion_tokens
                )

        logger.debug(f"Tokenizer loaded for model: {self.model}")
        return tokenizer
