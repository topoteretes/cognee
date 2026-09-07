from uuid import UUID
from typing import Optional, Any, List, Union
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from pydantic.alias_generators import to_camel
from cognee.modules.search.models.EvidenceReference import EvidenceReference
from cognee.modules.search.types.ContextFormat import ContextFormat
from cognee.modules.search.types.SearchType import SearchType


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
    evidence: List[EvidenceReference] = Field(default_factory=list)

    # TODO: Add return_type info
    search_type: SearchType
    only_context: bool = False

    # The query this payload answers. Carried so the prompt envelope can report how the
    # question was framed around the context instead of leaving the caller to guess.
    question: Optional[str] = None

    # Shape of the only_context result. CONTEXT (default) returns the bare context, as
    # it always has; PROMPT returns the whole envelope a completion would have received.
    # Typed as the enum so an invalid value cannot be stored and echoed back.
    context_format: ContextFormat = ContextFormat.CONTEXT
    session_context: Optional[str] = None
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None

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
    def prompt_envelope(self) -> dict:
        """Everything a completion would have been sent, as one dict.

        The question is included because that is the discrepancy this shape exists to
        close: a bare context leaves the caller guessing how cognee framed the question
        around it.
        """
        return {
            "question": self.question,
            "context": self.context,
            "session_context": self.session_context or "",
            "user_prompt": self.user_prompt,
            "system_prompt": self.system_prompt,
        }

    @property
    def result(self) -> Any:
        """Function used to determine search_result for users request.
        Return context if only_context is True, else return completion if it exists, else return result_object."""
        if self.only_context:
            if self.context_format == ContextFormat.PROMPT:
                return self.prompt_envelope
            return self.context
        elif self.completion:
            return self.completion
        elif self.context:
            return self.context
        else:
            return self.result_object
