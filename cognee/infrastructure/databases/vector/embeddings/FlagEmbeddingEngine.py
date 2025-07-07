import asyncio
from cognee.shared.logging_utils import get_logger
from typing import List, Optional
import numpy as np
import os
import aiohttp
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException
from cognee.infrastructure.llm.tokenizer.BAAI import FlagEmbeddingTokenizer

logger = get_logger("FlagEmbeddingEngine")


class FlagEmbeddingEngine(EmbeddingEngine):
    """
    Engine for embedding text using FlagEmbedding models, specifically designed for BAAI/bge-m3.

    This engine supports both local model loading and remote endpoint requests,
    similar to the implementation in core.py.

    Public methods:
    - embed_text: Embed a list of strings into vector representations.
    - get_vector_size: Retrieve the size of the embedding vectors.
    - get_tokenizer: Load the appropriate tokenizer for the specified model.
    """

    def __init__(
        self,
        model: Optional[str] = "BAAI/bge-m3",
        dimensions: Optional[int] = 1024,
        max_tokens: int = 512,
        endpoint: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model = model or "BAAI/bge-m3"
        self.dimensions = dimensions or 1024
        self.max_tokens = max_tokens
        self.endpoint = endpoint or os.getenv("EMBEDDING_ENDPOINT")
        self.batch_size = batch_size
        self.tokenizer = self.get_tokenizer()
        
        # Determine whether to use local model or remote endpoint
        self.use_endpoint = self.endpoint is not None
        
        if not self.use_endpoint:
            # Try to import and initialize local model
            try:
                from FlagEmbedding import BGEM3FlagModel
                self.embedding_model = BGEM3FlagModel(
                    self.model,
                    use_fp16=True,
                    devices=["cuda:0"] if os.environ.get("CUDA_VISIBLE_DEVICES") else ["cpu"],
                )
                logger.info(f"Initialized local FlagEmbedding model: {self.model}")
            except ImportError:
                logger.warning("FlagEmbedding not installed, falling back to endpoint mode")
                self.use_endpoint = True
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}, falling back to endpoint mode")
                self.use_endpoint = True

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings into vector representations.

        This method supports both local model inference and remote endpoint requests.
        It handles batching for large inputs and provides mock responses when enabled.

        Parameters:
        -----------
            - text (List[str]): A list of strings to be embedded.

        Returns:
        --------
            - List[List[float]]: A list of vectors representing the embedded texts.
        """
        try:
            if self.mock:
                logger.debug("Using mock embeddings")
                return [[0.0] * self.dimensions for _ in text]
            
            if self.use_endpoint:
                return await self._embed_via_endpoint(text)
            else:
                return await self._embed_via_local_model(text)

        except Exception as error:
            logger.error(f"Embedding error in FlagEmbeddingEngine: {str(error)}")
            raise EmbeddingException(f"Failed to index data points using model {self.model}")

    async def _embed_via_endpoint(self, text: List[str]) -> List[List[float]]:
        """Embed text using remote endpoint (similar to core.py implementation)"""
        if not self.endpoint:
            raise EmbeddingException("No endpoint configured for FlagEmbedding")
        
        async def process_batch(session, batch: List[str]) -> List[List[float]]:
            data = {"inputs": batch}
            headers = {"Content-Type": "application/json"}

            async with session.post(self.endpoint, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"HTTP error {response.status}: {error_text}")
                    raise EmbeddingException(f"Endpoint request failed: {error_text}")
                return await response.json()

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(0, len(text), self.batch_size):
                batch = text[i:i + self.batch_size]
                tasks.append(asyncio.create_task(process_batch(session, batch)))

            results = await asyncio.gather(*tasks)

        # Flatten results
        combined_responses = [item for sublist in results for item in sublist]
        return combined_responses

    async def _embed_via_local_model(self, text: List[str]) -> List[List[float]]:
        """Embed text using local FlagEmbedding model"""
        if not hasattr(self, 'embedding_model'):
            raise EmbeddingException("Local model not initialized")
        
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        def encode_batch(texts):
            results = []
            for text_item in texts:
                embedding = self.embedding_model.encode(text_item)
                if isinstance(embedding, dict) and 'dense_vecs' in embedding:
                    # Handle BGEM3 output format
                    dense_vec = embedding['dense_vecs']
                    if hasattr(dense_vec, 'tolist'):
                        results.append(dense_vec.tolist())
                    elif isinstance(dense_vec, list):
                        results.append(dense_vec)
                    else:
                        # Convert to list using numpy if available
                        try:
                            import numpy as np
                            results.append(np.array(dense_vec).tolist())
                        except ImportError:
                            results.append(list(dense_vec))
                else:
                    # Handle other formats
                    if hasattr(embedding, 'tolist'):
                        results.append(embedding.tolist())
                    elif isinstance(embedding, list):
                        results.append(embedding)
                    else:
                        # Convert to list using numpy if available
                        try:
                            import numpy as np
                            results.append(np.array(embedding).tolist())
                        except ImportError:
                            results.append(list(embedding))
            return results
        
        return await loop.run_in_executor(None, encode_batch, text)

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
        logger.debug(f"Loading tokenizer for FlagEmbeddingEngine with model: {self.model}")

        try:
            tokenizer = FlagEmbeddingTokenizer(model=self.model, max_tokens=self.max_tokens)
            logger.debug(f"Tokenizer loaded for FlagEmbeddingEngine: {self.model}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load FlagEmbedding tokenizer: {e}, using fallback")
            # Fallback to a simple tokenizer
            from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
            # return TikTokenTokenizer(model="gpt-4o", max_tokens=self.max_tokens)
