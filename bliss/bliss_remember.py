import asyncio
import json
from pathlib import Path

import cognee
from cognee.low_level import DataPoint

CUSTOM_PROMPT = """Your task is to extract the paper title, entities, and hypothesis from the paper below.
Each input uses the form Title: \"...\", content: ... For the paper, output only title plus entities and hypothesis. For each entity, output only name and kind; kind must be exactly "problem" or "module" (e.g. Problem A, alpha, beta). Do not extract math-style symbols such as X_A, Y_A, X_B, or Y_B.
For the hypothesis, output only text as one concise, testable claim about what the paper proposes—not a summary of methods or results—and restate explicit proposals directly or infer the implied claim, avoiding openers like "This study investigates...".
Do not set id, type, metadata, created_at, updated_at, or any other system fields."""


class Entity(DataPoint):
    name: str
    kind: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name", "kind"]}


class Hypothesis(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"], "identity_fields": ["text"]}


class Paper(DataPoint):
    """Root graph model passed to remember()."""

    title: str
    entities: list[Entity] = []
    hypothesis: Hypothesis | None = None
    metadata: dict = {"index_fields": ["title"], "identity_fields": ["title"]}


PAPERS_PATH = Path(__file__).parent / "data" / "papers.json"


def get_papers(path: Path = PAPERS_PATH) -> list[str]:
    """Return paper texts from the JSON corpus."""
    papers = json.loads(path.read_text())
    return [paper["text"] for paper in papers]


async def ingest_papers() -> None:
    """Remember all papers into the dataset."""
    await cognee.remember(
        get_papers(),
        graph_model=Paper,
        custom_prompt=CUSTOM_PROMPT,
        self_improvement=False,
    )


async def main() -> None:
    await cognee.forget(everything=True)
    await ingest_papers()


if __name__ == "__main__":
    asyncio.run(main())
