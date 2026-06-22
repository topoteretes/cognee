from typing import Annotated, List

from pydantic import Field

from cognee.infrastructure.engine import DataPoint, Dedup, Embeddable, LLMContext


class Skill(DataPoint):
    """A dataset-scoped procedural playbook loaded by the agentic retriever."""

    name: Annotated[str, Dedup()]
    description: str = ""
    procedure: str = ""
    declared_tools: List[str] = Field(default_factory=list)

    # Publisher / provenance metadata. Surfaced in the SaaS UI so users can see
    # which company or team maintains a skill. Populated from SKILL.md frontmatter
    # (maintainer/company/author, maintainer_url/url, version, tags, license) and
    # falls back to empty values for skills authored before these fields existed.
    maintainer: str = ""
    maintainer_url: str = ""
    skill_version: str = ""
    tags: List[str] = Field(default_factory=list)
    license: str = ""
    source_repo_url: str = ""

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
