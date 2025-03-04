from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEvalAdapter(ABC):
    @abstractmethod
    async def evaluate_answers(
        self, data: List[Dict[str, Any]], evaluator_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        pass
