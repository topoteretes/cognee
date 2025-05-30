from cognee.shared.logging_utils import get_logger
from typing import List, Optional
from fastembed import TextEmbedding
import litellm
import os
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer

litellm.set_verbose = False
logger = get_logger("FastembedEmbeddingEngine")


class FastembedEmbeddingEngine(EmbeddingEngine):
    """
    Manages the embedding process using a specified model to generate text embeddings.

    Public methods:

    - embed_text
    - get_vector_size
    - get_tokenizer

    Instance variables:

    - model: The name of the embedding model.
    - dimensions: The dimensionality of the embeddings.
    - mock: A flag indicating whether to use mocking instead of the actual embedding model.
    - MAX_RETRIES: The maximum number of retries for embedding operations.
    """

    model: str
    dimensions: int
    mock: bool

    MAX_RETRIES = 5

    def __init__(
        self,
        model: Optional[str] = "openai/text-embedding-3-large",
        dimensions: Optional[int] = 3072,
        max_tokens: int = 512,
    ):
        self.model = model
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.tokenizer = self.get_tokenizer()
        # self.retry_count = 0
        self.embedding_model = TextEmbedding(model_name=model)

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Embed the given text into numerical vectors.

        This method generates embeddings for a list of text strings. If mocking is enabled, it
        returns zero vectors instead. It handles exceptions by logging the error and raising an
        `EmbeddingException` on failure.

        Parameters:
        -----------

            - text (List[str]): A list of strings to be embedded.

        Returns:
        --------

            - List[List[float]]: A list of embeddings, where each embedding is a list of floats
              representing the vector form of the input text.
        """
        try:
            if self.mock:
                return [[0.0] * self.dimensions for _ in text]
            else:
                embeddings = self.embedding_model.embed(
                    text,
                    batch_size=len(text),
                    parallel=None,
                )

                return list(embeddings)

        except Exception as error:
            logger.error(f"Embedding error in FastembedEmbeddingEngine: {str(error)}")
            raise EmbeddingException(f"Failed to index data points using model {self.model}")

    def get_vector_size(self) -> int:
        """
        Return the size of the embedding vector produced by this engine.

        Returns:
        --------

            - int: The dimensionality of the embedding vectors.
        """
        return self.dimensions

    def get_tokenizer(self):
        """
        Instantiate and return the tokenizer used for preparing text for embedding.

        Returns:
        --------

            A tokenizer object configured for the specified model and maximum token size.
        """
        logger.debug("Loading tokenizer for FastembedEmbeddingEngine...")

        tokenizer = TikTokenTokenizer(model="gpt-4o", max_tokens=self.max_tokens)

        logger.debug("Tokenizer loaded for for FastembedEmbeddingEngine")
        return tokenizer
