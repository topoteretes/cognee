from enum import Enum
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

VALID_RATINGS = {"helpful", "harmful"}
MAX_CONTEXT_CONTENT_CHARS = 280
MIN_CANDIDATE_CONFIDENCE = 0.75


class ContextSection(str, Enum):
    """Sections under which active session-context guidance is grouped."""

    GOALS = "goals"
    RULES = "rules"
    PREFERENCES = "preferences"
    LESSONS_LEARNED = "lessons_learned"


VALID_SECTIONS = {section.value for section in ContextSection}


class ContextProfile(str, Enum):
    """Which kind of consumer a session-context lesson is for."""

    QA = "qa"
    AGENT = "agent"


VALID_PROFILES = {profile.value for profile in ContextProfile}


class AgentContextSection(str, Enum):
    """Sections for agent/tool-loop session-context guidance."""

    TOOL_RULES = "tool_rules"
    WORKFLOW_STATE = "workflow_state"
    SUCCESS_PATTERNS = "success_patterns"
    FAILURE_LESSONS = "failure_lessons"
    ENVIRONMENT_FACTS = "environment_facts"


AGENT_VALID_SECTIONS = {section.value for section in AgentContextSection}

SECTIONS_BY_PROFILE = {
    ContextProfile.QA.value: VALID_SECTIONS,
    ContextProfile.AGENT.value: AGENT_VALID_SECTIONS,
}


def valid_sections_for(profile: str) -> set:
    """Allowed section names for a context profile; empty set for an unknown profile."""
    return SECTIONS_BY_PROFILE.get(profile, set())


def normalize_content(text: str) -> str:
    """Deterministic exact-match key: lowercased, whitespace-collapsed, stripped."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return " ".join(text.strip().lower().split())


class ServedContextRating(BaseModel):
    """A single rating of a session-context entry that was served to the previous answer."""

    entry_id: str
    rating: str

    @field_validator("entry_id")
    @classmethod
    def entry_id_non_empty(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("entry_id must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("entry_id must be a non-empty string")
        return stripped

    @field_validator("rating")
    @classmethod
    def rating_valid(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("rating must be a string")
        normalized = v.strip().lower()
        if normalized not in VALID_RATINGS:
            raise ValueError(f"rating must be one of {sorted(VALID_RATINGS)}")
        return normalized


class CandidateContextUpdate(BaseModel):
    """A proposed new active session-context guidance entry emitted by feedback detection."""

    section: str = Field(
        description=(
            "One of goals, rules, preferences, or lessons_learned. Choose by whether the "
            "content is an objective, a constraint, presentation guidance, or durable knowledge."
        ),
    )
    content: str = Field(
        description=(
            "One short sentence containing only reusable session guidance, not the full user "
            "message or a one-off request."
        ),
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence from 0 to 1. Only candidates with confidence >= 0.75 are stored.",
    )

    @field_validator("section")
    @classmethod
    def section_valid(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("section must be a string")
        normalized = v.strip().lower()
        if normalized not in VALID_SECTIONS:
            raise ValueError(f"section must be one of {sorted(VALID_SECTIONS)}")
        return normalized

    @field_validator("content")
    @classmethod
    def content_truncate(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("content must be a string")
        stripped = v.strip()
        return stripped[:MAX_CONTEXT_CONTENT_CHARS]

    @field_validator("confidence")
    @classmethod
    def confidence_clamp(cls, v: float) -> float:
        if v is None:
            return 0.0
        try:
            value = float(v)
        except (TypeError, ValueError):
            raise ValueError("confidence must be a number")
        return max(0.0, min(1.0, value))


class CandidateGoalUpdate(CandidateContextUpdate):
    """Candidate update for the user's broader session objective."""

    section: Literal["goals"] = Field(
        default=ContextSection.GOALS.value,
        description="Fixed section value for goal updates.",
    )
    content: str = Field(
        description=(
            "The user's broader objective or desired outcome for the session. Use this for where "
            "the user is trying to get to, not constraints, facts, or answer format."
        ),
    )


class CandidateRuleUpdate(CandidateContextUpdate):
    """Candidate update for substantive instructions future answers must follow."""

    section: Literal["rules"] = Field(
        default=ContextSection.RULES.value,
        description="Fixed section value for rule updates.",
    )
    content: str = Field(
        description=(
            "An instruction future answers must obey: a constraint, requirement, assumption, "
            "source boundary, decision criterion, technical choice, or thing to avoid. Use rules "
            "for statements that change answer substance, even if phrased as facts."
        ),
    )


class CandidatePreferenceUpdate(CandidateContextUpdate):
    """Candidate update for presentation and communication preferences."""

    section: Literal["preferences"] = Field(
        default=ContextSection.PREFERENCES.value,
        description="Fixed section value for preference updates.",
    )
    content: str = Field(
        description=(
            "Presentation guidance only: style, tone, format, length, structure, terminology, "
            "examples, ordering, or level of detail. Use preferences only when the substantive "
            "answer would stay the same."
        ),
    )


class CandidateLessonLearnedUpdate(CandidateContextUpdate):
    """Candidate update for session knowledge future answers should reason from."""

    section: Literal["lessons_learned"] = Field(
        default=ContextSection.LESSONS_LEARNED.value,
        description="Fixed section value for lesson updates.",
    )
    content: str = Field(
        description=(
            "Durable knowledge future answers should reason from: a correction, clarification, "
            "discovered fact, cause, outcome, or prior assumption update. It may overlap with a "
            "rule when it captures reusable context behind that rule, but it must not be only a "
            "direct instruction or required action."
        ),
    )


CandidateContextUpdateVariant = Annotated[
    CandidateGoalUpdate
    | CandidateRuleUpdate
    | CandidatePreferenceUpdate
    | CandidateLessonLearnedUpdate,
    Field(discriminator="section"),
]


class AgentCandidateContextUpdate(CandidateContextUpdate):
    """A proposed new agent-profile session-context lesson (tool/workflow guidance)."""

    context_profile: Literal["agent"] = ContextProfile.AGENT.value
    section: str = Field(
        description=(
            "One of tool_rules, workflow_state, success_patterns, failure_lessons, or "
            "environment_facts. Choose by what kind of tool/workflow lesson the content is."
        ),
    )

    @field_validator("section")
    @classmethod
    def section_valid(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("section must be a string")
        normalized = v.strip().lower()
        if normalized not in AGENT_VALID_SECTIONS:
            raise ValueError(f"section must be one of {sorted(AGENT_VALID_SECTIONS)}")
        return normalized


class CandidateToolRuleUpdate(AgentCandidateContextUpdate):
    """Candidate update for a reusable tool constraint or safe-invocation pattern."""

    section: Literal["tool_rules"] = Field(
        default=AgentContextSection.TOOL_RULES.value,
        description="Fixed section value for tool-rule updates.",
    )
    content: str = Field(
        description=(
            "A reusable constraint or safe/required way to invoke a tool — what to do or avoid "
            "when using it next time."
        ),
    )


class CandidateWorkflowStateUpdate(AgentCandidateContextUpdate):
    """Candidate update for where the current task stands."""

    section: Literal["workflow_state"] = Field(
        default=AgentContextSection.WORKFLOW_STATE.value,
        description="Fixed section value for workflow-state updates.",
    )
    content: str = Field(
        description=(
            "Where the current task stands and what must survive context compaction to continue "
            "it — progress, the next step, or a pending decision."
        ),
    )


class CandidateSuccessPatternUpdate(AgentCandidateContextUpdate):
    """Candidate update for an approach worth repeating."""

    section: Literal["success_patterns"] = Field(
        default=AgentContextSection.SUCCESS_PATTERNS.value,
        description="Fixed section value for success-pattern updates.",
    )
    content: str = Field(
        description="An approach or sequence that solved the user's request and is worth repeating.",
    )


class CandidateFailureLessonUpdate(AgentCandidateContextUpdate):
    """Candidate update for something to avoid next time."""

    section: Literal["failure_lessons"] = Field(
        default=AgentContextSection.FAILURE_LESSONS.value,
        description="Fixed section value for failure-lesson updates.",
    )
    content: str = Field(
        description=(
            "An error, retry, or rejected path and what to avoid or do differently next time."
        ),
    )


class CandidateEnvironmentFactUpdate(AgentCandidateContextUpdate):
    """Candidate update for a discovered environment fact."""

    section: Literal["environment_facts"] = Field(
        default=AgentContextSection.ENVIRONMENT_FACTS.value,
        description="Fixed section value for environment-fact updates.",
    )
    content: str = Field(
        description=(
            "A discovered fact about the repository, runtime, configuration, or environment that "
            "future steps need."
        ),
    )


AgentCandidateContextUpdateVariant = Annotated[
    CandidateToolRuleUpdate
    | CandidateWorkflowStateUpdate
    | CandidateSuccessPatternUpdate
    | CandidateFailureLessonUpdate
    | CandidateEnvironmentFactUpdate,
    Field(discriminator="section"),
]


class AgentContextExtraction(BaseModel):
    """LLM output for the batch pass: agent-profile lessons drawn from trace evidence.

    ``lessons`` is intentionally typed as the plain base model, not the section-specific
    discriminated union (``AgentCandidateContextUpdateVariant``). instructor's structured-output
    modes ask the model to pick one of N pydantic classes via a discriminator field before the
    fields even validate -- reliable for large/cloud models, but empirically unreliable for
    smaller local models (measured against a local Ollama `llama3.1:8b`, both with instructor's
    default `json_mode` and with `json_schema_mode`): the model frequently omits the discriminator
    field entirely or returns a shape `union_tag_not_found` can't resolve, so most extractions are
    rejected outright. The base model enforces the identical constraint
    (``AgentCandidateContextUpdate.section_valid`` already restricts ``section`` to
    ``AGENT_VALID_SECTIONS``) without requiring a discriminator match, and the only real consumer
    of this output (``extract_batch_agent_context`` -> ``apply_candidate_updates`` ->
    ``_coerce_candidate_model``) accepts any ``CandidateContextUpdate`` subclass via `isinstance`,
    so nothing downstream depends on the specific subclass identity -- only on ``section`` already
    being valid, which the base model guarantees on its own. In local testing against
    `llama3.1:8b`, the discriminated union was rejected outright on most trials while this
    base-model schema was consistently accepted -- and, since it asks the model to fill in fewer
    structural fields, also faster. This does loosen the JSON schema instructor emits for every
    provider (``section`` becomes a free-form string instead of an enumerated discriminator), but
    ``section_valid`` remains the actual enforcement point regardless of provider, so correctness
    is unaffected -- an invalid ``section`` is still rejected, just via the validator instead of
    the schema.
    """

    lessons: List[AgentCandidateContextUpdate] = Field(
        default_factory=list,
        description=(
            "Reusable agent/tool lessons drawn from the traces. Each item's `section` must be "
            "one of tool_rules, workflow_state, success_patterns, failure_lessons, or "
            "environment_facts, matching the kind of tool/workflow lesson the content is."
        ),
    )

    @field_validator("lessons", mode="before")
    @classmethod
    def normalize_lessons(cls, value):
        if not isinstance(value, list):
            return []
        normalized = []
        for item in value:
            if isinstance(item, AgentCandidateContextUpdate):
                item = item.model_dump()
            if isinstance(item, dict):
                item = dict(item)
                section = item.get("section")
                if isinstance(section, str):
                    item["section"] = section.strip().lower()
                item.setdefault("context_profile", ContextProfile.AGENT.value)
            normalized.append(item)
        return normalized


class SessionContextEntry(BaseModel):
    """A stored active session-context lesson, tagged by profile (qa or agent)."""

    id: str
    section: str
    context_profile: str = ContextProfile.QA.value
    content: str
    normalized_content: str = ""
    confidence: float = 0.0
    created_at: str
    source_feedback_ids: List[str] = Field(default_factory=list)
    source_trace_ids: List[str] = Field(default_factory=list)
    helpful_count: int = 0
    harmful_count: int = 0
    priority: int = 0
    last_served_at: Optional[str] = None
    embedding: Optional[List[float]] = None
    kind: Literal["context"] = "context"

    @field_validator("id")
    @classmethod
    def id_non_empty(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("id must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("id must be a non-empty string")
        return stripped

    @field_validator("section")
    @classmethod
    def section_normalize(cls, v: str) -> str:
        # Membership is profile-dependent, so only normalize here; the (profile, section)
        # pair is validated in validate_section_for_profile once both fields are set.
        if not isinstance(v, str):
            raise ValueError("section must be a string")
        return v.strip().lower()

    @field_validator("context_profile")
    @classmethod
    def context_profile_valid(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("context_profile must be a string")
        normalized = v.strip().lower()
        if normalized not in VALID_PROFILES:
            raise ValueError(f"context_profile must be one of {sorted(VALID_PROFILES)}")
        return normalized

    @field_validator("content")
    @classmethod
    def content_non_empty_truncate(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("content must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("content must be a non-empty string")
        return stripped[:MAX_CONTEXT_CONTENT_CHARS]

    @field_validator("normalized_content")
    @classmethod
    def normalized_content_string(cls, v: str) -> str:
        if v is None:
            return ""
        if not isinstance(v, str):
            raise ValueError("normalized_content must be a string")
        return normalize_content(v)

    @field_validator("confidence")
    @classmethod
    def confidence_clamp(cls, v: float) -> float:
        if v is None:
            return 0.0
        try:
            value = float(v)
        except (TypeError, ValueError):
            raise ValueError("confidence must be a number")
        return max(0.0, min(1.0, value))

    @field_validator("helpful_count", "harmful_count")
    @classmethod
    def counts_non_negative(cls, v: int) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("count must be an integer")
        if v < 0:
            raise ValueError("count must be >= 0")
        return v

    @field_validator("source_feedback_ids", "source_trace_ids")
    @classmethod
    def source_id_lists_only_strings(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("source id list must be a list")
        normalized = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("source id list must contain only strings")
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized

    @model_validator(mode="after")
    def derive_normalized_content(self):
        if not self.normalized_content:
            self.normalized_content = normalize_content(self.content)
        return self

    @model_validator(mode="after")
    def validate_section_for_profile(self):
        allowed = valid_sections_for(self.context_profile)
        if self.section not in allowed:
            raise ValueError(
                f"section '{self.section}' is not valid for profile "
                f"'{self.context_profile}'; allowed: {sorted(allowed)}"
            )
        return self


class SessionFeedbackEntry(BaseModel):
    """A stored feedback record describing how a turn rated/extended session context."""

    id: str
    created_at: str
    raw_text: str
    referenced_qa_ids: List[str] = Field(default_factory=list)
    influencing_context_ids: List[str] = Field(default_factory=list)
    candidate_context_entries: List[dict] = Field(default_factory=list)
    kind: Literal["feedback"] = "feedback"

    @field_validator("id")
    @classmethod
    def id_non_empty(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("id must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("id must be a non-empty string")
        return stripped

    @field_validator("referenced_qa_ids", "influencing_context_ids")
    @classmethod
    def id_lists_only_strings(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("id list must be a list")
        normalized = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("id list must contain only strings")
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized
