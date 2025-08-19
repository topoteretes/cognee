from abc import ABC, abstractmethod
from typing import Any


class BaseFeedback(ABC):
    """Base class for all user feedback operations."""

    @abstractmethod
    async def add_feedback(self, feedback_text: str) -> Any:
        """Add user feedback to the system."""
        pass
