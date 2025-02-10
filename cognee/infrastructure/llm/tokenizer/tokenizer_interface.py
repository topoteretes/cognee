from typing import List, Protocol, Any
from abc import abstractmethod


class TokenizerInterface(Protocol):
    """Tokenizer interface"""

    @abstractmethod
    def extract_tokens(self, text: str) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def decode_single_token(self, token: int) -> str:
        raise NotImplementedError
