from typing import Any, Dict, List, Optional
from cognee.low_level import DataPoint


class ToolCall(DataPoint):
    """A single tool invocation within a skill run."""

    tool_name: str
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    success: bool = True
    duration_ms: int = 0
    metadata: dict = {"index_fields": []}


class CandidateSkill(DataPoint):
    """A skill considered during routing, with its retrieval score and signals."""

    skill_id: str
    score: float = 0.0
    signals: Optional[Dict[str, Any]] = None
    metadata: dict = {"index_fields": []}


class SkillRun(DataPoint):
    """Record of a skill execution within a session."""

    run_id: str
    session_id: str
    cognee_session_id: str = ""
    task_text: str
    result_summary: str = ""
    success_score: float = 0.0  # 0.0 to 1.0

    # Routing decision
    candidate_skills: List[CandidateSkill] = []
    selected_skill: Optional["Skill"] = None
    selected_skill_id: str = ""
    task_pattern_id: str = ""
    router_version: str = ""

    tool_trace: List[ToolCall] = []

    error_type: str = ""
    error_message: str = ""

    started_at_ms: int = 0
    latency_ms: int = 0
    feedback: float = 0.0  # -1.0 to 1.0, 0 = no feedback

    cache_qa_id: str = ""  # QA entry id in SessionManager cache (for promotion cleanup)

    previous_run: Optional["SkillRun"] = None

    metadata: dict = {"index_fields": ["task_text", "result_summary"]}


from .skill import Skill  # noqa: E402

SkillRun.model_rebuild()
