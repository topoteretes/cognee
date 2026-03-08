from typing import Dict, List, Optional
from pydantic import BaseModel, field_validator


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
    """

    time: str
    question: str
    context: str
    answer: str
    qa_id: Optional[str] = None
    feedback_text: Optional[str] = None
    feedback_score: Optional[int] = None
    used_graph_element_ids: Optional[Dict[str, List[str]]] = None

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
