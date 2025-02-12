from abc import ABC, abstractmethod
from typing import List
from cognee.modules.pipelines.tasks.Task import Task


class BaseTaskGetter(ABC):
    """Abstract base class for asynchronous task retrieval implementations."""

    @abstractmethod
    async def get_tasks(self) -> List[Task]:
        """Asynchronously retrieve a list of tasks. Must be implemented by subclasses."""
        pass
