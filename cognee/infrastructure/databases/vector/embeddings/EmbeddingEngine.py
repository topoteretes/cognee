from typing import Protocol


class EmbeddingEngine(Protocol):
    """
    Defines an interface for embedding text. Provides methods to embed text and get the
    vector size.
    """

    async def embed_text(self, text: list[str]) -> list[list[float]]:
        """
        Embed the provided text and return a list of embedded vectors.

        Parameters:
        -----------

            - text (list[str]): A list of strings representing the text to be embedded.

        Returns:
        --------

            - list[list[float]]: A list of lists, where each sublist contains the encoded
              representation of the corresponding text input.
        """
        raise NotImplementedError()

    def get_vector_size(self) -> int:
        """
        Retrieve the size of the embedding vector.

        Returns:
        --------

            - int: An integer representing the number of dimensions in the embedding vector.
        """
        raise NotImplementedError()

    def get_batch_size(self) -> int:
        """
        Return the desired batch size for embedding calls

        Returns:

        """
        raise NotImplementedError()
