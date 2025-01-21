from typing import List, Protocol, Any
from abc import abstractmethod


class TokenizerInterface(Protocol):
    """Tokenizer interface"""

    @abstractmethod
    def extract_tokens(self, text: str) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def num_tokens_from_text(self, text: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def trim_text_to_max_tokens(self, text: str) -> str:
        raise NotImplementedError
