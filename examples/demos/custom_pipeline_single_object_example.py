"""
Custom pipeline example: LLM-powered entity extraction on DataPoint objects.

Demonstrates the deferred-call pipeline pattern (TaskSpec / BoundTask)
with typed DataPoint models, field annotations, LLM structured output,
and per-source freshness tracking via source_content_hash.

Usage:
    uv run python examples/demos/custom_pipeline_single_object_example.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

from cognee.infrastructure.engine import DataPoint, Embeddable, Dedup
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.pipelines.tasks.task import task
from cognee.modules.pipelines.operations.run_pipeline import run_pipeline
from cognee.tasks.storage import add_data_points


# -- Data models --


class ScientificClaim(DataPoint):
    """A factual claim extracted from text."""

    text: Annotated[str, Embeddable("Claim text for semantic search"), Dedup()]
    subject: str = ""
    confidence: float = 1.0


class Person(DataPoint):
    """A person mentioned in the text."""

    name: Annotated[str, Embeddable("Person name"), Dedup()]
    role: str = ""
    claims: Optional[List[ScientificClaim]] = None


class AnalysisResult(BaseModel):
    """LLM output model for structured extraction."""

    people: List[Person] = Field(default_factory=list)
    claims: List[ScientificClaim] = Field(default_factory=list)


# -- Pipeline tasks --


@task
async def extract_entities(text: str) -> AnalysisResult:
    """Use LLM to extract people and claims from text."""
    result = await LLMGateway.acreate_structured_output(
        text_input=text,
        system_prompt=(
            "Extract all people and scientific claims from the text. "
            "For each person, provide their name and role. "
            "For each claim, provide the claim text, subject, and confidence (0-1)."
        ),
        response_model=AnalysisResult,
    )
    return result


@task
async def link_claims_to_people(analysis: AnalysisResult) -> List[Person]:
    """Associate claims with the people who made them, using LLM."""

    class ClaimAssignment(BaseModel):
        person_name: str
        claim_texts: List[str]

    class Assignments(BaseModel):
        assignments: List[ClaimAssignment]

    assignments = await LLMGateway.acreate_structured_output(
        text_input=(
            f"People: {[p.name for p in analysis.people]}\n"
            f"Claims: {[c.text for c in analysis.claims]}"
        ),
        system_prompt=(
            "Assign each claim to the person who made it or is most associated with it. "
            "Return a list of assignments, each with a person_name and their claim_texts."
        ),
        response_model=Assignments,
    )

    # Build lookup and attach claims to people
    claim_lookup = {c.text: c for c in analysis.claims}
    for assignment in assignments.assignments:
        for person in analysis.people:
            if person.name.lower() == assignment.person_name.lower():
                person.claims = [
                    claim_lookup[t] for t in assignment.claim_texts if t in claim_lookup
                ]

    return analysis.people


@task
async def store_and_summarize(people: List[Person]) -> str:
    """Store DataPoints in graph + vector DBs, then return a summary."""

    # add_data_points persists nodes and edges to graph DB,
    # and indexes embeddable fields in vector DB
    await add_data_points(people)

    lines = []
    for person in people:
        # source_content_hash is stamped by the pipeline provenance system;
        # it carries the content hash of the source document this node came from
        hash_display = person.source_content_hash or "N/A"
        lines.append(f"{person.name} ({person.role}) [source_hash: {hash_display[:12]}]")
        if person.claims:
            for claim in person.claims:
                lines.append(f"  - {claim.text} [confidence: {claim.confidence}]")
        else:
            lines.append("  (no claims linked)")
    return "\n".join(lines)


# -- Run --


async def main():
    import cognee
    from cognee.infrastructure.databases.relational.create_db_and_tables import (
        create_db_and_tables,
    )

    await create_db_and_tables()

    # Clean slate
    await cognee.forget(everything=True)

    sample_text = (
        "Albert Einstein published the theory of general relativity in 1915, "
        "describing gravity as spacetime curvature. Marie Curie discovered "
        "polonium and radium, winning Nobel Prizes in both physics and chemistry. "
        "Niels Bohr proposed the atomic model with quantized electron orbits in 1913."
    )

    # Run the custom pipeline
    results = await run_pipeline(
        [
            extract_entities(),
            link_claims_to_people(),
            store_and_summarize(),
        ],
        data=sample_text,
        pipeline_name="entity_extraction",
    )

    print(results[0] if results else "No output")

    # Search the graph
    print("\n--- Search: 'Who worked on gravity?' ---")
    answer = await cognee.search(
        "Who worked on gravity?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
    )
    print(f"  {answer}")

    # Clean up
    print("\n--- Forget everything ---")
    result = await cognee.forget(everything=True)
    print(f"  {result}")


if __name__ == "__main__":
    asyncio.run(main())
