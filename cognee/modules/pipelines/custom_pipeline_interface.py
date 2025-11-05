from typing import Protocol, Any
from abc import abstractmethod


class CustomPipelineInterface(Protocol):
    """
    Defines an interface for creating and running a custom pipeline.
    """

    @abstractmethod
    async def run_pipeline(self) -> Any:
        raise NotImplementedError
