"""Settle a conflict that conflict review cannot decide, and see recall use the choice.

Two sources disagree on where Acme was founded. improve(review_conflicts=True) stores the
disagreement as a conflict, but a founding city never changes, so the review cannot pick a
winner and leaves the conflict unresolved. cognee.list_conflicts() shows the competing facts
with their ids; cognee.resolve_conflict() records which one the user accepts. Recall is
printed before and after the decision, and a second review shows the decision is kept.

Requires: LLM_API_KEY. Conflict review needs EDGE_EVIDENCE_ENABLED=true (the default).
Run: uv run python examples/guides/resolve_conflicts.py
"""

import asyncio
from uuid import UUID

import cognee
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.search.types import SearchType
from cognee.tasks.ingestion.data_item import DataItem

DATASET = "resolve_conflicts"
QUESTION = "Where was Acme founded?"


async def main():
    try:
        await cognee.forget(dataset=DATASET)
    except DatasetNotFoundError:
        pass  # First run: the dataset does not exist yet.

    # 1. Two sources that disagree on a fact that cannot change over time.
    for label, text in [
        ("Company registry", "Acme was founded in Berlin."),
        ("Conference speaker bio", "Acme was founded in Munich."),
    ]:
        result = await cognee.remember(
            DataItem(text, label=label, literal_text=True),
            dataset_name=DATASET,
            self_improvement=False,
        )
    dataset_id = UUID(str(result.dataset_id))

    # 2. Conflict review records the disagreement but cannot settle it.
    await cognee.improve(dataset=DATASET, review_conflicts=True)
    conflict = (await cognee.list_conflicts(dataset_id, status="unresolved"))[0]
    print("Unresolved conflict:", conflict.attribute, [fact.text for fact in conflict.facts])
    before = await cognee.recall(
        QUESTION, query_type=SearchType.GRAPH_COMPLETION, datasets=[DATASET]
    )
    print("Before the decision:", before[0].text)

    # 3. The user accepts the registry's fact. The id must be one of the conflict's facts.
    berlin = next(fact for fact in conflict.facts if "Berlin" in fact.text)
    resolved = await cognee.resolve_conflict(
        conflict.id,
        dataset_id=dataset_id,
        selected_fact_id=berlin.id,
        reason="The company registry is the authoritative source.",
    )
    print("Resolved:", resolved.status, "-", resolved.selected_fact)
    after = await cognee.recall(
        QUESTION, query_type=SearchType.GRAPH_COMPLETION, datasets=[DATASET]
    )
    print("After the decision:", after[0].text)

    # 4. A later automatic review keeps the user's decision.
    await cognee.improve(dataset=DATASET, review_conflicts=True)
    kept = (await cognee.list_conflicts(dataset_id))[0]
    print("After another review:", kept.status, "-", kept.selected_fact)


if __name__ == "__main__":
    asyncio.run(main())
