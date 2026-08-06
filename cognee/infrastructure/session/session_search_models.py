from functools import lru_cache
from typing import Any, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    CandidateContextUpdateVariant,
    ServedContextRating,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel | str)
EvidenceStatus = Literal["pending", "deferred", "completed", "failed"]


def _normalize_evidence(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    evidence = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item:
            evidence.append(item[:MAX_CONTEXT_CONTENT_CHARS])
        if len(evidence) == 3:
            break
    return evidence


class SessionSearchCompletion(BaseModel, Generic[ResponseT]):
    """One foreground answer plus the evidence needed for later maintenance."""

    response: ResponseT | str
    is_acknowledgement: bool = False
    feedback_evidence: list[str] = Field(default_factory=list)
    future_context_evidence: list[str] = Field(default_factory=list)

    @field_validator("feedback_evidence", "future_context_evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: Any) -> list[str]:
        return _normalize_evidence(value)

    @model_validator(mode="after")
    def validate_response_kind(self):
        model_args = type(self).__pydantic_generic_metadata__.get("args", ())
        if not model_args:
            raise ValueError("SessionSearchCompletion must specify a response type")

        expected_type = model_args[0]
        if self.is_acknowledgement:
            if not isinstance(self.response, str):
                raise ValueError("acknowledgement responses must be strings")
        elif expected_type is not str and not isinstance(self.response, expected_type):
            raise ValueError("normal responses must match the configured response type")
        return self


@lru_cache(maxsize=32)
def get_session_search_completion_model(
    response_model: type[ResponseT],
) -> type[SessionSearchCompletion[ResponseT]]:
    """Return the validated wrapper type for a public response model."""
    if response_model is not str and not (
        isinstance(response_model, type) and issubclass(response_model, BaseModel)
    ):
        raise TypeError("response_model must be str or a Pydantic model")
    return cast(type[SessionSearchCompletion[ResponseT]], SessionSearchCompletion[response_model])


class SessionTurnSnapshot(BaseModel):
    """Immutable session state used by one latency-optimized turn."""

    model_config = ConfigDict(frozen=True)

    raw_message: str
    recent_qas: tuple[tuple[str, str, str], ...] = ()
    completion_history: str = ""
    active_context: str = ""
    active_context_ids: tuple[str, ...] = ()
    previous_qa_id: str | None = None
    previous_question: str | None = None
    previous_answer: str | None = None
    previous_served_context: tuple[tuple[str, str], ...] = ()


class SessionTurnEvidence(BaseModel):
    """Persisted evidence consumed by maintenance or later distillation."""

    id: str
    created_at: str
    dataset_id: str | None = None
    current_qa_id: str | None = None
    current_raw_message: str
    current_response: str
    previous_qa_id: str | None = None
    previous_question: str | None = None
    previous_answer: str | None = None
    previous_served_context: tuple[tuple[str, str], ...] = ()
    feedback_evidence: list[str] = Field(default_factory=list)
    future_context_evidence: list[str] = Field(default_factory=list)
    status: EvidenceStatus = "pending"
    error: str | None = None
    distilled_at: str | None = None
    kind: Literal["turn_evidence"] = "turn_evidence"

    @field_validator("feedback_evidence", "future_context_evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: Any) -> list[str]:
        return _normalize_evidence(value)

    @field_validator("error", mode="before")
    @classmethod
    def bound_error(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(value).strip()[:MAX_CONTEXT_CONTENT_CHARS] or None


class SessionMaintenanceWorkItem(BaseModel):
    """Immutable identity handed to process-local maintenance."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    user_id: str
    session_id: str
    dataset_id: str | None = None
    trace_id: str | None = None


class SessionMaintenanceResult(BaseModel):
    """Validated ratings and context candidates generated from turn evidence."""

    served_context_ratings: list[ServedContextRating] = Field(default_factory=list)
    candidate_context_updates: list[CandidateContextUpdateVariant] = Field(default_factory=list)

    @field_validator("served_context_ratings", "candidate_context_updates", mode="after")
    @classmethod
    def cap_three(cls, value: list) -> list:
        return value[:3]


class MaintenanceApplyResult(BaseModel):
    """Outcome of applying one maintenance result."""

    model_config = ConfigDict(frozen=True)

    applied_ids: tuple[str, ...] = ()
    skipped_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @field_validator("errors", mode="before")
    @classmethod
    def bound_errors(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(error).strip()[:MAX_CONTEXT_CONTENT_CHARS] for error in value[:3])
