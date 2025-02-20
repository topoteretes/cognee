from abc import ABC, abstractmethod
from typing import List, Type
from cognee.modules.pipelines.tasks.Task import Task


class BaseTaskGetter(ABC):
    """Abstract base class for asynchronous task retrieval implementations."""

    @abstractmethod
    async def get_tasks(self, chunk_size: int, chunker: Type) -> List[Task]:
        """Asynchronously retrieve a list of tasks. Must be implemented by subclasses."""
        pass
