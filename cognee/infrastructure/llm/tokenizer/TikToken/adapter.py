from typing import List, Any
import tiktoken

from ..tokenizer_interface import TokenizerInterface


class TikTokenTokenizer(TokenizerInterface):
    """
    Tokenizer adapter for OpenAI.
    Inteded to be used as part of LLM Embedding and LLM Adapters classes
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 8191,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # Initialize TikToken for GPT based on model
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def extract_tokens(self, text: str) -> List[Any]:
        # Using TikToken's method to tokenize text
        token_ids = self.tokenizer.encode(text)
        return token_ids

    def decode_token_list(self, tokens: List[Any]) -> List[Any]:
        if not isinstance(tokens, list):
            tokens = [tokens]
        return [self.tokenizer.decode(i) for i in tokens]

    def decode_single_token(self, token: int):
        return self.tokenizer.decode_single_token_bytes(token).decode("utf-8", errors="replace")

    def count_tokens(self, text: str) -> int:
        """
        Returns the number of tokens in the given text.
        Args:
            text: str

        Returns:
            number of tokens in the given text

        """
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

    def trim_text_to_max_tokens(self, text: str) -> str:
        """
        Trims the text so that the number of tokens does not exceed max_tokens.

        Args:
        text (str): Original text string to be trimmed.

        Returns:
        str: Trimmed version of text or original text if under the limit.
        """
        # First check the number of tokens
        num_tokens = self.count_tokens(text)

        # If the number of tokens is within the limit, return the text as is
        if num_tokens <= self.max_tokens:
            return text

        # If the number exceeds the limit, trim the text
        # This is a simple trim, it may cut words in half; consider using word boundaries for a cleaner cut
        encoded_text = self.tokenizer.encode(text)
        trimmed_encoded_text = encoded_text[: self.max_tokens]
        # Decoding the trimmed text
        trimmed_text = self.tokenizer.decode(trimmed_encoded_text)
        return trimmed_text
