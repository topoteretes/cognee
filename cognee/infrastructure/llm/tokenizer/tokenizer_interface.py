from typing import List, Protocol, Any
from abc import abstractmethod


class TokenizerInterface(Protocol):
    """
    Defines an interface for tokenizers that provides methods for token extraction,
    counting, and decoding.
    """

    @abstractmethod
    def extract_tokens(self, text: str) -> List[Any]:
        """
        Extract tokens from the given text.

        Parameters:
        -----------

            - text (str): The input text from which to extract tokens.
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Parameters:
        -----------

            - text (str): The input text for which to count tokens.

        Returns:
        --------

            - int: The total count of tokens in the input text.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_single_token(self, token: int) -> str:
        """
        Decode a single token represented by an integer into its string representation.

        Parameters:
        -----------

            - token (int): The integer representation of the token to decode.

        Returns:
        --------

            - str: The string representation of the decoded token.
        """
        raise NotImplementedError
