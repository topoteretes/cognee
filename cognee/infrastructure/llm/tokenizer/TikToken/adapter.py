from typing import List, Any, Optional
import tiktoken

from ..tokenizer_interface import TokenizerInterface


class TikTokenTokenizer(TokenizerInterface):
    """
    Tokenizer adapter for OpenAI. Intended to be used as part of LLM Embedding and LLM
    Adapters classes.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_completion_tokens: int = 8191,
    ):
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        # Initialize TikToken for GPT based on model
        if model:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        else:
            # Use default if model not provided
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def extract_tokens(self, text: str) -> List[Any]:
        """
        Extract tokens from the given text.

        Parameters:
        -----------

            - text (str): The text to be tokenized.

        Returns:
        --------

            - List[Any]: A list of token IDs representing the encoded text.
        """
        # Using TikToken's method to tokenize text
        token_ids = self.tokenizer.encode(text)
        return token_ids

    def decode_token_list(self, tokens: List[Any]) -> List[Any]:
        """
        Decode a list of token IDs back into their corresponding text representations.

        Parameters:
        -----------

            - tokens (List[Any]): A list of token IDs to be decoded.

        Returns:
        --------

            - List[Any]: A list of decoded text representations of the tokens.
        """
        if not isinstance(tokens, list):
            tokens = [tokens]
        return [self.tokenizer.decode(i) for i in tokens]

    def decode_single_token(self, token: int):
        """
        Decode a single token ID into its corresponding text representation.

        Parameters:
        -----------

            - token (int): A single token ID to be decoded.

        Returns:
        --------

            The decoded text representation of the token.
        """
        return self.tokenizer.decode_single_token_bytes(token).decode("utf-8", errors="replace")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Parameters:
        -----------

            - text (str): The text for which to count the tokens.

        Returns:
        --------

            - int: The number of tokens in the given text.
        """
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

    def trim_text_to_max_completion_tokens(self, text: str) -> str:
        """
        Trim the text so that the number of tokens does not exceed max_completion_tokens.

        Parameters:
        -----------

            - text (str): Original text string to be trimmed.

        Returns:
        --------

            - str: Trimmed version of text or original text if under the limit.
        """
        # First check the number of tokens
        num_tokens = self.count_tokens(text)

        # If the number of tokens is within the limit, return the text as is
        if num_tokens <= self.max_completion_tokens:
            return text

        # If the number exceeds the limit, trim the text
        # This is a simple trim, it may cut words in half; consider using word boundaries for a cleaner cut
        encoded_text = self.tokenizer.encode(text)
        trimmed_encoded_text = encoded_text[: self.max_completion_tokens]
        # Decoding the trimmed text
        trimmed_text = self.tokenizer.decode(trimmed_encoded_text)
        return trimmed_text
