"""
Custom pipeline example: LLM-powered entity extraction on DataPoint objects.

Demonstrates a custom Task pipeline with typed DataPoint models, field
annotations, LLM structured output, and per-source freshness tracking via
source_content_hash — run against a named dataset so the nodes it stores are
attributed to that dataset and searchable with recall().

Usage:
    uv run python examples/demos/custom_pipelines/custom_pipeline_single_object_example.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

import cognee
from cognee.infrastructure.engine import DataPoint, Dedup, Embeddable
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.models import Data
from cognee.modules.pipelines import Task
from cognee.tasks.storage import add_data_points

DATASET_NAME = "science_claims"

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
    claims: list[ScientificClaim] | None = None


class AnalysisResult(DataPoint):
    """LLM output model for structured extraction.

    A DataPoint rather than a plain BaseModel so the pipeline's provenance
    stamping walks into it and reaches the nested Person and ScientificClaim
    objects; a plain BaseModel would stop the walk and leave their
    source_content_hash empty. It is never stored — only its contents are.
    """

    people: list[Person] = Field(default_factory=list)
    claims: list[ScientificClaim] = Field(default_factory=list)


# -- Pipeline tasks --


async def extract_entities(data_items: list[Data]) -> AnalysisResult:
    """Read the ingested document(s) and use LLM to extract people and claims."""
    text_parts = []
    for data_item in data_items:
        async with open_data_file(data_item.raw_data_location, mode="r", encoding="utf-8") as file:
            text_parts.append(file.read())

    result = await LLMGateway.acreate_structured_output(
        text_input="\n".join(text_parts),
        system_prompt=(
            "Extract all people and scientific claims from the text. "
            "For each person, provide their name and role. "
            "For each claim, provide the claim text, subject, and confidence (0-1)."
        ),
        response_model=AnalysisResult,
    )
    return result


async def link_claims_to_people(analysis: AnalysisResult) -> list[Person]:
    """Associate claims with the people who made them, using LLM."""

    class ClaimAssignment(BaseModel):
        person_name: str
        claim_texts: list[str]

    class Assignments(BaseModel):
        assignments: list[ClaimAssignment]

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


async def store_and_summarize(people: list[Person]) -> str:
    """Store DataPoints in graph + vector DBs, then print and return a summary."""

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

    summary = "\n".join(lines)
    print(summary)
    return summary


# -- Run --


async def main():
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

    # Ingest the text into a dataset first. This creates the dataset, stores the
    # text as a Data record with a content hash, and is what makes the graph the
    # custom pipeline builds below both attributable and searchable.
    await cognee.add(sample_text, dataset_name=DATASET_NAME)

    # Run the custom pipeline over the dataset's ingested documents. With no
    # `data` argument the first task receives the dataset's Data records.
    await cognee.run_custom_pipeline(
        tasks=[
            Task(extract_entities),
            Task(link_claims_to_people),
            Task(store_and_summarize),
        ],
        dataset=DATASET_NAME,
        pipeline_name="entity_extraction",
    )

    # Recall from the graph
    print("\n--- Recall: 'Who worked on gravity?' ---")
    answer = await cognee.recall(
        "Who worked on gravity?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print(f"  {answer}")

    # Clean up
    print("\n--- Forget everything ---")
    result = await cognee.forget(everything=True)
    print(f"  {result}")


if __name__ == "__main__":
    asyncio.run(main())
