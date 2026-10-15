from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(max_length=200)
    version: str = Field(max_length=200)
    text: str = Field(min_length=1, max_length=16000)


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event_id: UUID
    dataset_id: UUID
    captured_at: AwareDatetime
    consent_version: Literal["answer-evaluation-v1"]
    prompt: str = Field(min_length=1, max_length=8000)
    answer: str = Field(min_length=1, max_length=16000)
    context: str = Field(default="", max_length=16000)
    evidence: list[Evidence] = Field(default_factory=list, max_length=20)
    model: str = Field(min_length=1, max_length=200)
    plugin: Literal["claude", "codex"]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    recall_budget_seconds: Literal[45] = 45

    @model_validator(mode="after")
    def bounded(self):
        age = (datetime.now(timezone.utc) - self.captured_at).total_seconds()
        if age < -300 or age > 30 * 86400:
            raise ValueError("Trace outside retention window")
        if len(self.model_dump_json().encode()) > 64000:
            raise ValueError("Trace exceeds 64 KB")
        return self


def score(better, worse, total):
    return round(47 + 53 * max(0, min(1, (better - worse) / total))) if total else 47
