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

    # The two messages an only_context call stands in for, kept apart as the LLM receives
    # them: user_prompt is the conversation history, the question and the retrieval
    # context rendered through the retriever's template, and the session guidance block;
    # system_prompt is the retriever's task template. Set only when only_context is on,
    # the retriever sends one templated prompt, and retrieval found something; otherwise
    # both are None and `result` falls back to `context`.
    user_prompt: str | None = None
    system_prompt: str | None = None

    dataset_name: str | None = None
    dataset_id: UUID | None = None
    dataset_tenant_id: UUID | None = None

    # Set when this dataset could not be searched (empty graph, missing
    # collection, unresolvable code seed). The result fields are empty and the
    # search's list still carries one entry per dataset, so a caller can tell
    # "nothing matched here" from "this dataset was not searched" -- and read why.
    error: str | None = None

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

        With only_context, return the user prompt when one was built (the system prompt
        travels separately, see ``system_prompt``), else the bare context; otherwise
        return the completion if it exists, else the result_object.
        """
        if self.only_context:
            return self.user_prompt if self.user_prompt is not None else self.context
        elif self.completion:
            return self.completion
        elif self.context:
            return self.context
        else:
            return self.result_object
