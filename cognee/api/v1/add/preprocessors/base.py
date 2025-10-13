"""
Base classes for the cognee add preprocessor system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, BinaryIO
from pydantic import BaseModel

from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class PreprocessorContext(BaseModel):
    """Context passed to preprocessors during processing."""

    model_config = {"arbitrary_types_allowed": True}

    data: Union[BinaryIO, List[BinaryIO], str, List[str]]
    dataset_name: str
    user: Optional[User] = None
    node_set: Optional[List[str]] = None
    vector_db_config: Optional[Dict] = None
    graph_db_config: Optional[Dict] = None
    dataset_id: Optional[str] = None
    preferred_loaders: Optional[List[str]] = None
    incremental_loading: bool = True
    extra_params: Dict[str, Any] = {}


class PreprocessorResult(BaseModel):
    """Result returned by preprocessors."""

    modified_context: Optional[PreprocessorContext] = None
    stop_processing: bool = False
    error: Optional[str] = None


class Preprocessor(ABC):
    """
    Base class for all cognee add preprocessors.

    Preprocessors can modify the processing context, add custom logic,
    or handle specific data types before main pipeline processing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this preprocessor."""
        pass

    @abstractmethod
    def can_handle(self, context: PreprocessorContext) -> bool:
        """
        Check if this preprocessor can handle the given context.

        Args:
            context: The current processing context

        Returns:
            True if this preprocessor should process this context
        """
        pass

    @abstractmethod
    async def process(self, context: PreprocessorContext) -> PreprocessorResult:
        """
        Process the given context.

        Args:
            context: The current processing context

        Returns:
            PreprocessorResult with any modifications or errors
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}"


class PreprocessorRegistry:
    """Registry for managing and executing preprocessors."""

    def __init__(self):
        self._preprocessors: List[Preprocessor] = []

    def register(self, preprocessor: Preprocessor) -> None:
        """Register a preprocessor."""
        if not isinstance(preprocessor, Preprocessor):
            raise TypeError(
                f"Preprocessor must inherit from Preprocessor, got {type(preprocessor)}"
            )

        if any(prep.name == preprocessor.name for prep in self._preprocessors):
            raise ValueError(f"Preprocessor with name '{preprocessor.name}' already registered")

        self._preprocessors.append(preprocessor)

    def unregister(self, name: str) -> bool:
        """Unregister a preprocessor by name."""
        for i, prep in enumerate(self._preprocessors):
            if prep.name == name:
                del self._preprocessors[i]
                return True
        return False

    def get_preprocessors(self) -> List[Preprocessor]:
        """Get all registered preprocessors ordered by priority."""
        return self._preprocessors.copy()

    def get_applicable_preprocessors(self, context: PreprocessorContext) -> List[Preprocessor]:
        """Get preprocessors that can handle the given context."""
        return [prep for prep in self._preprocessors if prep.can_handle(context)]

    async def process_with_selected_preprocessors(
        self, context: PreprocessorContext, preprocessor_names: List[str]
    ) -> PreprocessorContext:
        """
        Process context through only the specified preprocessors.

        Args:
            context: The initial context
            preprocessor_names: List of preprocessor names to run

        Returns:
            The final processed context

        Raises:
            Exception: If any preprocessor encounters an error or preprocessor name not found
        """
        current_context = context

        selected_preprocessors: List[Preprocessor] = []
        for name in preprocessor_names:
            preprocessor = next((prep for prep in self._preprocessors if prep.name == name), None)
            if preprocessor is None:
                available_names = [prep.name for prep in self._preprocessors]
                raise ValueError(
                    f"Preprocessor '{name}' not found. Available preprocessors: {available_names}"
                )
            selected_preprocessors.append(preprocessor)

        for preprocessor in selected_preprocessors:
            if not preprocessor.can_handle(current_context):
                logger.warning(
                    f"Preprocessor '{preprocessor.name}' cannot handle current context, skipping"
                )
                continue

            try:
                result = await preprocessor.process(current_context)

                if result.error:
                    raise Exception(f"Preprocessor '{preprocessor.name}' failed: {result.error}")

                if result.modified_context:
                    current_context = result.modified_context

                if result.stop_processing:
                    break

            except Exception as e:
                raise Exception(
                    f"Preprocessor '{preprocessor.name}' encountered an error: {str(e)}"
                ) from e

        return current_context
