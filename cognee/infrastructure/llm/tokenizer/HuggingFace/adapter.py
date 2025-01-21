from typing import List, Any

from ..tokenizer_interface import TokenizerInterface


class HuggingFaceTokenizer(TokenizerInterface):
    def __init__(
        self,
        model: str,
        max_tokens: int = float("inf"),
    ):
        self.model = model
        self.max_tokens = max_tokens

    def extract_tokens(self, text: str) -> List[Any]:
        raise NotImplementedError

    def num_tokens_from_text(self, text: str) -> int:
        raise NotImplementedError

    def trim_text_to_max_tokens(self, text: str) -> str:
        raise NotImplementedError
