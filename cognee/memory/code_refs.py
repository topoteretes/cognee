"""The code a memory entry is about, named exactly: which declaration, at which version.

An agent that read source before writing a note can say so with one
``CodeReference`` per declaration it inspected: enola's fact id of the
declaration (``enola locate file:line``), the version of its code the agent
saw (``body_hash``) and the revision it read. The reference is written by the
caller and stored as given; cognee never infers one from the note's prose, so a
later reader finds and re-checks the exact code without an LLM naming it.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CodeReference(BaseModel):
    """One declaration a memory entry is about, at the version its author saw."""

    repo: str = Field(min_length=1, description="The repository label enola writes on the fact.")
    enola_id: str = Field(
        pattern=r"^[0-9a-f]{32}$",
        description="enola's fact id of the declaration (facts.jsonl ``id``; ``enola locate``).",
    )
    name: str = Field(
        min_length=1, description="The declaration's qualified name, as enola names it."
    )
    file: str = Field(min_length=1, description="Repository-relative path of the declaration.")
    line: int = Field(gt=0, description="Its declaration line at the observed revision.")
    body_hash: str = Field(
        min_length=1, description="enola's version of the declaration's code (enola >= 0.4.37)."
    )
    decl_hash: str | None = Field(
        default=None,
        description=(
            "Tells apart declarations sharing an id, such as a getter and its setter "
            "(enola >= 0.4.38)."
        ),
    )
    observed_commit: str = Field(
        min_length=1, description="The revision the author read the code at."
    )
    extractor_version: str = Field(
        min_length=1, description="The enola version that produced the id and hashes."
    )
    resolution: Literal["exact", "enclosing", "ambiguous"] = Field(
        default="exact",
        description=(
            "What the reference names: the declaration itself, only the declaration "
            "enclosing the cited lines (a nested function enola does not extract), or "
            "one of several candidates."
        ),
    )
