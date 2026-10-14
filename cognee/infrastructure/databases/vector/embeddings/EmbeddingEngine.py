from typing import Protocol, TypeAlias

# One embedding: the model's vector for one input text.
Vector: TypeAlias = list[float]


class EmbeddingResult(list[Vector]):
    """What an engine's embed_text() returns: a plain list with one vector per input
    text, which also carries the input tokens the provider consumed in ``tokens``
    (embeddings have no output tokens).

    Callers that only want the vectors use it as the list it is, exactly as before
    token counting existed; EmbeddingGateway reads ``tokens`` to record usage. An
    engine that returns a plain list reports 0 tokens. Anything that builds a new
    list from it (slicing, ``+``, ``list()``) is a plain list again, so an engine
    combining partial results must wrap them in a new EmbeddingResult.
    """

    def __init__(self, vectors=(), tokens: int = 0):
        super().__init__(vectors)
        self.tokens = tokens


def embedding_tokens(vectors: list[Vector]) -> int:
    """The input tokens an embed_text() result carries; 0 for a plain list."""
    return getattr(vectors, "tokens", 0)


class EmbeddingEngine(Protocol):
    """
    Defines an interface for embedding text. Provides methods to embed text and get the
    vector size.
    """

    async def embed_text(self, text: list[str]) -> EmbeddingResult:
        """
        Embed the provided text and report the input tokens it consumed.

        Callers may use the engine directly and get a list of vectors; embedding
        through EmbeddingGateway also records the tokens.

        Parameters:
        -----------

            - text (list[str]): A list of strings representing the text to be embedded.

        Returns:
        --------

            - EmbeddingResult: One vector per input text, with the token count in
              ``tokens``.
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
