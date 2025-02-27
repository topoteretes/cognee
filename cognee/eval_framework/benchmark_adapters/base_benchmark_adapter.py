from abc import ABC, abstractmethod
from typing import List, Optional


class BaseBenchmarkAdapter(ABC):
    @abstractmethod
    def load_corpus(
        self, limit: Optional[int] = None, seed: int = 42, load_golden_context: bool = False
    ) -> List[str]:
        pass
