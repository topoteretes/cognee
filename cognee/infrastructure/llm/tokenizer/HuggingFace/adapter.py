from typing import List, Any

from ..tokenizer_interface import TokenizerInterface


class HuggingFaceTokenizer(TokenizerInterface):
    def __init__(
        self,
        model: str,
        max_tokens: int = 512,
    ):
        self.model = model
        self.max_tokens = max_tokens

        # Import here to make it an optional dependency
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def extract_tokens(self, text: str) -> List[Any]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def count_tokens(self, text: str) -> int:
        """
        Returns the number of tokens in the given text.
        Args:
            text: str

        Returns:
            number of tokens in the given text

        """
        return len(self.tokenizer.tokenize(text))

    def decode_single_token(self, encoding: int):
        # HuggingFace tokenizer doesn't have the option to decode tokens
        raise NotImplementedError
