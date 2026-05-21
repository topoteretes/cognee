from typing import Annotated, List

from pydantic import Field

from cognee.infrastructure.engine import DataPoint, Dedup, Embeddable, LLMContext


class Skill(DataPoint):
    """A dataset-scoped procedural playbook loaded by the agentic retriever."""

    name: Annotated[str, Dedup()]
    description: str = ""
    procedure: str = ""
    declared_tools: List[str] = Field(default_factory=list)

    source_file: str = ""
    source_dir: str = ""
    content_hash: str = ""
    dataset_scope: List[str] = Field(default_factory=list)
    is_active: bool = True

    skill_text: str = ""
    search_text: Annotated[str, Embeddable(), LLMContext()] = ""

    metadata: dict = Field(
        default={
            "index_fields": ["search_text"],
            "identity_fields": ["name", "source_dir"],
        }
    )
