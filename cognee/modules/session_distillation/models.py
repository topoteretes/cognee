"""Data models for session distillation.

Distillation turns the high-quality learnings of one finished session into a
human-readable document that gets cognified into the persistent knowledge graph.
The pipeline is: gate -> novelty search -> curator (1 LLM call) -> anchoring
search -> writers (one LLM call per surviving lesson) -> render -> add + cognify.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Quality gate: a context entry is distillable only when it was never rated harmful,
# its net helpfulness is not negative, and its confidence clears this threshold.
MIN_GATE_CONFIDENCE = 0.75

# How many existing-knowledge snippets the novelty search collects per gated entry.
NOVELTY_SNIPPETS_PER_ENTRY = 3
# How many existing entity names the anchoring search collects per surviving lesson.
GLOSSARY_ENTITIES_PER_LESSON = 5
# Session digest: question truncation length for the one-line-per-turn overview.
MAX_DIGEST_QUESTION_CHARS = 150
# Truncation for the per-entry source user message shown to the curator (provenance).
MAX_SOURCE_MESSAGE_CHARS = 240


class CuratedLesson(BaseModel):
    """One merged, deduplicated lesson proposed by the curator call."""

    working_statement: str = Field(
        description=(
            "The lesson as one standalone sentence, merging all member entries that "
            "express the same underlying learning."
        )
    )
    member_entry_ids: List[str] = Field(
        description="Ids of the gated session-context entries folded into this lesson.",
    )
    kind: Literal["domain_fact", "working_practice"] = Field(
        description=(
            "domain_fact: knowledge about the subject matter itself. "
            "working_practice: guidance about how to work on it."
        ),
    )
    novelty: Literal["new", "already_known"] = Field(
        description=(
            "already_known when the provided existing-knowledge snippets show the graph "
            "already contains this lesson; new otherwise."
        ),
    )
    overlap_note: Optional[str] = Field(
        default=None,
        description="If the lesson touches existing knowledge without duplicating it, how.",
    )


class CurationPlan(BaseModel):
    """Output of the single curator call: what survives distillation and how it is grouped."""

    session_summary: str = Field(
        description="Two or three sentences describing what this session was about.",
    )
    lessons: List[CuratedLesson] = Field(
        default_factory=list,
        description=(
            "Durable lessons worth persisting beyond this session. Session-local trivia "
            "(file paths, one-off requests, transient state) must be omitted entirely."
        ),
    )


class DistilledLesson(BaseModel):
    """Output of one writer call: a lesson rewritten for knowledge-graph extraction."""

    statement: str = Field(
        description=(
            "The lesson as standalone, context-free prose. Entity names from the provided "
            "glossary must be used verbatim where they apply."
        )
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Glossary entity names actually used in the statement.",
    )
    why_learned: str = Field(
        default="",
        description="One sentence, grounded in the evidence, on how this was learned.",
    )


class DistillationResult(BaseModel):
    """Outcome of one distill_session call."""

    session_id: str
    dataset_id: Optional[str] = None
    status: Literal[
        "completed",
        "no_gated_entries",
        "nothing_new",
        "curator_failed",
        "no_lessons_written",
    ]
    documents: List[str] = Field(default_factory=list)
    session_summary: Optional[str] = None
    gated_entry_count: int = 0
    lesson_count: int = 0
    skipped_already_known: int = 0
