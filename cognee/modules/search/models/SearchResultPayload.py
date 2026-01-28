from uuid import UUID
from typing import Optional, Any, List, Union
from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic.alias_generators import to_camel
from cognee.modules.search.types.SearchType import SearchType


class SearchResultPayload(BaseModel):
    """Result payload from retriever classes."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    result_object: Any = None
    context: Optional[str] = None
    completion: Optional[Union[str, List[str]]] = None

    # TODO: Add return_type info
    search_type: SearchType
    only_context: bool = False

    dataset_name: Optional[str] = None
    dataset_id: Optional[UUID] = None
    dataset_tenant_id: Optional[UUID] = None

    @field_serializer("result_object")
    def serialize_complex_types(self, v: Any):
        """
        Custom serializer to handle complex types in result_object.
        Transforms non-JSON-compatible types to their string representation.
        """

        # Helper to check if a value is a "simple" JSON-compatible type
        def is_simple(item):
            return isinstance(item, (int, float, dict, str, bool, type(None)))

        # Handle Lists
        if isinstance(v, list):
            return [item if is_simple(item) else str(item) for item in v]

        # Handle Dictionaries
        if isinstance(v, dict):
            return {key: (val if is_simple(val) else str(val)) for key, val in v.items()}

        # Fallback for the object itself
        return v if is_simple(v) else str(v)

    @property
    def result(self) -> Any:
        """Function used to determine search_result for users request.
        Return context if only_context is True, else return completion if it exists, else return result_object."""
        if self.only_context:
            return self.context
        elif self.completion:
            return self.completion
        else:
            return self.result_object
