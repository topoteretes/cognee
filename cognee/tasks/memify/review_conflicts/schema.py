"""The structured response one review call asks the LLM for.

Every field here is a label the call rendered — ``n1`` for a subject, ``r2`` or
``s3`` for a fact, ``f1`` for a stored conflict. The validator resolves them
back to ids and rejects any label the call did not show.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ReportedConflict(BaseModel):
    about: str
    attribute: str
    kind: Literal["time_varying", "fixed"]
    text: str
    current: list[str] = Field(default_factory=list)
    superseded: list[str] = Field(default_factory=list)
    conflicting: list[str] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    reviewed: list[str]
    conflicts: list[ReportedConflict] = Field(default_factory=list)
    drop: list[str] = Field(default_factory=list)
