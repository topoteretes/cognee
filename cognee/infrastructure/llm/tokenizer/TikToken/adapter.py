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
        max_tokens: int = float("inf"),
    ):
        self.model = model
        self.max_tokens = max_tokens
        # Initialize TikToken for GPT based on model
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def extract_tokens(self, text: str) -> List[Any]:
        tokens = []
        # Using TikToken's method to tokenize text
        token_ids = self.tokenizer.encode(text)
        # Go through tokens and decode them to text value
        for token_id in token_ids:
            token = self.tokenizer.decode([token_id])
            tokens.append(token)
        return tokens

    def num_tokens_from_text(self, text: str) -> int:
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
        num_tokens = self.num_tokens_from_string(text)

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
