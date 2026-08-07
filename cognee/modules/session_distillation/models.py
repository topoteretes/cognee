"""Data models and tunables for session distillation.

Distillation turns a finished session's distillable context entries into standalone lesson
documents in the knowledge graph. The flow:

    select -> batch (qa + candidates) -> curate per batch -> judge + write per lesson -> persist

Curator calls run in parallel by batch; judge/write calls run in parallel by lesson.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# -- Tunables ----------------------------------------------------------------

# Gate: a context entry is distillable only when it was never rated harmful and its
# confidence clears this threshold. Deterministic, no search/LLM.
MIN_GATE_CONFIDENCE = 0.75

# Batching: pack capped timeline blocks into coarse batches. Six worst-case QA
# blocks plus separators stays below the budget, so the batching logic can stay simple.
BATCH_CHAR_BUDGET = 16_000
CURATOR_BLOCKS_PER_BATCH = 6
MAX_QA_QUESTION_CHARS = 1_200
MAX_QA_ANSWER_CHARS = 1_200
MAX_CANDIDATE_CHARS = 280

# Bounded concurrency for the two parallel fan-outs.
CURATOR_CONCURRENCY = 5
WRITER_CONCURRENCY = 5

# Per-lesson search limits.
NOVELTY_LESSONS_PER_LESSON = 5  # similar previously-persisted lessons, for already_known
GLOSSARY_ENTITIES_PER_LESSON = 20  # existing entity names, for verbatim anchoring


# -- Curator output (one call per batch) -------------------------------------


class ProposedLesson(BaseModel):
    """One durable lesson the curator proposes from a session batch."""

    working_statement: str = Field(
        description="One standalone sentence capturing the durable learning."
    )
    member_entry_ids: List[str] = Field(
        default_factory=list,
        description="Ids of the candidate memories this lesson draws from (may be empty).",
    )


class CuratorBatchOutput(BaseModel):
    """Proposed lessons from one curator batch call."""

    lessons: List[ProposedLesson] = Field(default_factory=list)


# -- Writer/rejecter output (one call per proposed lesson) -------------------


class WrittenLesson(BaseModel):
    """A per-lesson decision: accept (and write it) or reject (with a reason)."""

    accept: bool = Field(description="True to persist this lesson, False to drop it.")
    reason: Optional[Literal["already_known", "not_durable", "unsupported"]] = Field(
        default=None,
        description="Why the lesson was rejected, when accept is False.",
    )
    statement: str = Field(
        default="",
        description="Standalone, entity-anchored prose for the lesson, when accepted.",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Glossary entity names used in the statement.",
    )
    why_learned: str = Field(
        default="",
        description="One sentence naming the situation it was learned in.",
    )


# -- Result ------------------------------------------------------------------


class DistillationResult(BaseModel):
    """Outcome of one distill_session call."""

    session_id: str
    dataset_id: Optional[str] = None
    status: Literal[
        "completed",
        "no_gated_entries",
        "no_proposed_lessons",
        "no_accepted_lessons",
    ]
    documents: List[str] = Field(default_factory=list)
