from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer
from pydantic.alias_generators import to_camel

from cognee.modules.search.models.EvidenceReference import EvidenceReference
from cognee.modules.search.types.SearchType import SearchType


class SearchResultPayload(BaseModel):
    """Result payload from retriever classes."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    result_object: Any = None
    context: str | list[str] | None = None
    # NOTE: dict must precede BaseModel in the union so a plain dict validates
    # as-is instead of being coerced into an empty bare BaseModel.
    completion: str | list[str] | list[dict] | dict | BaseModel | list[BaseModel] | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)

    # TODO: Add return_type info
    search_type: SearchType
    only_context: bool = False

    # The full LLM input an only_context call stands in for: the system prompt (session
    # guidance, conversation history, task template) and the rendered user prompt, as one
    # string. Set only when only_context is on, the retriever sends one templated prompt,
    # and retrieval found something; otherwise None and `result` falls back to `context`.
    prompt: str | None = None

    dataset_name: str | None = None
    dataset_id: UUID | None = None
    dataset_tenant_id: UUID | None = None

    @field_serializer("result_object")
    def serialize_complex_types(self, v: Any):
        """
        Custom serializer to handle complex types in result_object.
        Transforms non-JSON-compatible types to their string representation.
        """

        # Helper to check if a value is a "simple" JSON-compatible type
        def is_simple(item):
            return isinstance(item, (int, float, dict, str, bool, type(None)))

        if isinstance(v, list) and all(isinstance(item, dict) for item in v):
            # Handle List of Dictionaries
            return [
                {key: (val if is_simple(val) else str(val)) for key, val in item.items()}
                for item in v
            ]
        elif isinstance(v, list):
            # Handle Lists
            return [item if is_simple(item) else str(item) for item in v]
        elif isinstance(v, dict):
            # Handle Dictionaries
            return {key: (val if is_simple(val) else str(val)) for key, val in v.items()}
        else:
            # Fallback for the object itself
            return v if is_simple(v) else str(v)

    @field_serializer("completion")
    def serialize_completion(self, v: Any):
        """Serialize completion field. Supports str, list, dict, and Pydantic BaseModel."""
        if v is None:
            return None
        if isinstance(v, BaseModel):
            return v.model_dump()
        if isinstance(v, list):
            return [item.model_dump() if isinstance(item, BaseModel) else item for item in v]
        return v

    @property
    def result(self) -> Any:
        """Function used to determine search_result for users request.

        With only_context, return the full LLM input when one was built, else the bare
        context; otherwise return the completion if it exists, else the result_object.
        """
        if self.only_context:
            return self.prompt if self.prompt is not None else self.context
        elif self.completion:
            return self.completion
        elif self.context:
            return self.context
        else:
            return self.result_object
