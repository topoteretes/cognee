from typing import List, Any

from ..tokenizer_interface import TokenizerInterface


class HuggingFaceTokenizer(TokenizerInterface):
    """
    Implements a tokenizer using the Hugging Face Transformers library.

    Public methods include:
    - extract_tokens
    - count_tokens
    - decode_single_token

    Instance variables include:
    - model: str
    - max_completion_tokens: int
    - tokenizer: AutoTokenizer
    """

    def __init__(
        self,
        model: str,
        max_completion_tokens: int = 512,
    ):
        self.model = model
        self.max_completion_tokens = max_completion_tokens

        # Import here to make it an optional dependency
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def extract_tokens(self, text: str) -> List[Any]:
        """
        Extract tokens from the given text using the tokenizer.

        Parameters:
        -----------

            - text (str): The input text to be tokenized.

        Returns:
        --------

            - List[Any]: A list of tokens extracted from the input text.
        """
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Parameters:
        -----------

            - text (str): The input text for which to count tokens.

        Returns:
        --------

            - int: The total number of tokens in the input text.
        """
        return len(self.tokenizer.tokenize(text))

    def decode_single_token(self, encoding: int):
        """
        Attempt to decode a single token from its encoding, which is not implemented in this
        tokenizer.

        Parameters:
        -----------

            - encoding (int): The integer encoding of the token to decode.
        """
        # HuggingFace tokenizer doesn't have the option to decode tokens
        raise NotImplementedError
