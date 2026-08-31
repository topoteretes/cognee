from uuid import UUID
from typing import Optional, Any, List, Union
from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic.alias_generators import to_camel
from cognee.modules.search.types.SearchType import SearchType


def serialize_value(val: Any) -> Any:
    """Recursively serialize complex types, BaseModels, UUIDs, sets, dicts to JSON-safe structures."""
    if isinstance(val, BaseModel):
        return serialize_value(val.model_dump())
    elif isinstance(val, UUID):
        return str(val)
    elif isinstance(val, (list, tuple, set)):
        return [serialize_value(item) for item in val]
    elif isinstance(val, dict):
        return {
            (str(key) if not isinstance(key, (str, int)) else key): serialize_value(value)
            for key, value in val.items()
        }
    elif isinstance(val, (int, float, str, bool, type(None))):
        return val
    else:
        return str(val)


class SearchResultPayload(BaseModel):
    """Result payload from retriever classes."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    result_object: Any = None
    context: Optional[Union[str, List[str]]] = None
    # NOTE: dict must precede BaseModel in the union so a plain dict validates
    # as-is instead of being coerced into an empty bare BaseModel.
    completion: Optional[Union[str, List[str], List[dict], dict, BaseModel, List[BaseModel]]] = None

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
        return serialize_value(v)

    @field_serializer("completion")
    def serialize_completion(self, v: Any):
        """Serialize completion field. Supports str, list, dict, and Pydantic BaseModel."""
        if v is None:
            return None
        return serialize_value(v)

    @property
    def result(self) -> Any:
        """Function used to determine search_result for users request.
        Return context if only_context is True, else return completion if it exists, else return result_object."""
        if self.only_context:
            return self.context
        elif self.completion is not None:
            return self.completion
        elif self.context is not None:
            return self.context
        else:
            return self.result_object
