from typing import List, Any

from transformers import AutoTokenizer

from ..tokenizer_interface import TokenizerInterface


class HuggingFaceTokenizer(TokenizerInterface):
    def __init__(
        self,
        model: str,
        max_tokens: int = 512,
    ):
        self.model = model
        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def extract_tokens(self, text: str) -> List[Any]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def num_tokens_from_text(self, text: str) -> int:
        """
        Returns the number of tokens in the given text.
        Args:
            text: str

        Returns:
            number of tokens in the given text

        """
        return len(self.tokenizer.tokenize(text))

    def trim_text_to_max_tokens(self, text: str) -> str:
        raise NotImplementedError
