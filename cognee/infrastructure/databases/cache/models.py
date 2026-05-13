from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from cognee.modules.agent_memory.sanitization import (
    MAX_SERIALIZED_VALUE_LENGTH,
    sanitize_value,
    truncate_text,
)


def _validate_list_of_str(value: object, key: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{key}[{i}] must be a string")
    return value


class SessionQAEntry(BaseModel):
    """
    Canonical format for a QA entry stored in session cache.

    Fields:
        time: ISO format timestamp when the QA was created.
        qa_id: Unique identifier for the entry (required for update/delete).
        question: User's question.
        context: Context used to generate the answer.
        answer: Generated answer.
        feedback_text: Optional user feedback text.
        feedback_score: Optional feedback score 1-5.
        used_graph_element_ids: Optional dict with only "node_ids" and "edge_ids" (lists of str).
        memify_metadata: Optional dict with memify status keys (e.g. "feedback_weights_applied") and bool values.
    """

    time: str
    question: str
    context: str
    answer: str
    qa_id: Optional[str] = None
    feedback_text: Optional[str] = None
    feedback_score: Optional[int] = None
    used_graph_element_ids: Optional[Dict[str, List[str]]] = None
    memify_metadata: Optional[Dict[str, bool]] = None

    @field_validator("used_graph_element_ids")
    @classmethod
    def used_graph_element_ids_only_node_and_edge_ids(
        cls, v: Optional[Dict[str, List[str]]]
    ) -> Optional[Dict[str, List[str]]]:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("used_graph_element_ids must be a dict or None")
        allowed = {"node_ids", "edge_ids"}
        if set(v.keys()) - allowed:
            raise ValueError("used_graph_element_ids may only have keys 'node_ids' and 'edge_ids'")
        out: Dict[str, List[str]] = {}
        if "node_ids" in v:
            out["node_ids"] = _validate_list_of_str(v["node_ids"], "node_ids")
        if "edge_ids" in v:
            out["edge_ids"] = _validate_list_of_str(v["edge_ids"], "edge_ids")
        return out if out else None

    @field_validator("feedback_score")
    @classmethod
    def feedback_score_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 5):
            raise ValueError("feedback_score must be between 1 and 5")
        return v

    @field_validator("memify_metadata")
    @classmethod
    def memify_metadata_only_pipeline_keys(
        cls, v: Optional[Dict[str, bool]]
    ) -> Optional[Dict[str, bool]]:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("memify_metadata must be a dict or None")
        out: Dict[str, bool] = {}
        for key, val in v.items():
            if not isinstance(key, str) or not isinstance(val, bool):
                raise ValueError("memify_metadata may only have string keys and bool values")
            out[key] = val
        return out if out else None


class SessionAgentTraceEntry(BaseModel):
    """
    Canonical format for one agent trace step stored in session cache.

    Fields:
        trace_id: Unique identifier for the trace step.
        origin_function: Agent method/function that produced the step.
        status: Execution status for the step.
        memory_query: Optional memory lookup query used by the step.
        memory_context: Optional memory context returned to the step.
        method_params: Serialized method parameters for the step.
        method_return_value: Serialized method return value for the step.
        error_message: Error details for failed steps.
        session_feedback: Per-step feedback generated from the step only.
    """

    trace_id: str
    origin_function: str
    status: str
    memory_query: str = ""
    memory_context: str = ""
    method_params: Dict[str, Any] = Field(default_factory=dict)
    method_return_value: Any = None
    error_message: str = ""
    session_feedback: str = ""

    @field_validator("trace_id", "origin_function", "status")
    @classmethod
    def required_non_empty_string(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("value must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("value must be a non-empty string")
        return truncate_text(stripped, MAX_SERIALIZED_VALUE_LENGTH)

    @field_validator("memory_query", "memory_context", "error_message", "session_feedback")
    @classmethod
    def normalize_optional_text_fields(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("value must be a string")
        return truncate_text(v.strip(), MAX_SERIALIZED_VALUE_LENGTH)

    @field_validator("method_params")
    @classmethod
    def sanitize_method_params(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("method_params must be a dict")
        sanitized = sanitize_value(v)
        if not isinstance(sanitized, dict):
            raise ValueError("method_params must sanitize to a dict")
        return sanitized

    @field_validator("method_return_value")
    @classmethod
    def sanitize_method_return_value(cls, v: Any) -> Any:
        return sanitize_value(v)
