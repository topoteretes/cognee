from dataclasses import dataclass
from typing import Protocol, TypeAlias

# One embedding: the model's vector for one input text.
Vector: TypeAlias = list[float]


@dataclass
class EmbeddingResult:
    """What an engine returns for one embed_text() call: a vector per input text and
    the input tokens the provider consumed (embeddings have no output tokens)."""

    vectors: list[Vector]
    tokens: int


class EmbeddingEngine(Protocol):
    """
    Defines an interface for embedding text. Provides methods to embed text and get the
    vector size.
    """

    async def embed_text(self, text: list[str]) -> EmbeddingResult:
        """
        Embed the provided text and report the input tokens it consumed.

        Called by EmbeddingGateway, which records the tokens; everything
        else embeds through its embed_text.

        Parameters:
        -----------

            - text (list[str]): A list of strings representing the text to be embedded.

        Returns:
        --------

            - EmbeddingResult: One vector per input text, plus the token count.
        """
        raise NotImplementedError("Subclasses must implement embed_text()")

    def get_vector_size(self) -> int:
        """
        Retrieve the size of the embedding vector.

        Returns:
        --------

            - int: An integer representing the number of dimensions in the embedding vector.
        """
        raise NotImplementedError("Subclasses must implement get_vector_size()")

    def get_batch_size(self) -> int:
        """
        Return the desired batch size for embedding calls

        Returns:

        """
        raise NotImplementedError("Subclasses must implement get_batch_size()")

    async def input_limit(self) -> int | None:
        """
        How many tokens of text the configured model accepts in one input, when the
        provider can tell (its model table, the loaded model, the server); None when
        it cannot. Async because a provider may have to ask its server. Resolved once
        per engine by ``resolve_input_limit`` (embeddings/input_limit.py), which sets
        ``max_completion_tokens`` to min(EMBEDDING_MAX_COMPLETION_TOKENS, this).

        Returns:
        --------

            - int | None: The model's input limit in tokens, or None when unknown.
        """
        raise NotImplementedError("Subclasses must implement input_limit()")
