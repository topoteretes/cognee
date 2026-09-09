from abc import ABC, abstractmethod
from typing import Any


class BaseEvalAdapter(ABC):
    @abstractmethod
    async def evaluate_answers(
        self, data: list[dict[str, Any]], evaluator_metrics: list[str]
    ) -> list[dict[str, Any]]:
        pass
