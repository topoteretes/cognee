from abc import ABC, abstractmethod
from typing import Any, Optional, Callable


class BaseFeedback(ABC):
    """Base class for all user feedback operations."""

    @abstractmethod
    async def add_feedback(self, feedback_text: str) -> Any:
        """Retrieves context based on the query."""
        pass
