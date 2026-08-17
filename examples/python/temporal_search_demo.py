"""Temporal search over timestamped agent events.

This example uses the lower-level ``add -> cognify -> search`` workflow to build
a temporal graph and query it with ``SearchType.TEMPORAL``. It also passes
``session_id`` values to show how separate agent interactions can query the same
timeline while keeping their conversational context distinct.

Usage:
    uv run python examples/python/temporal_search_demo.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio
from pprint import pprint
from typing import Any

import cognee
from cognee import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET_NAME = "temporal_agent_memory_demo"

EVENTS = [
    "2024-01-08 09:00 UTC: Session alpha opened an onboarding task for customer Nova.",
    "2024-01-08 11:30 UTC: Session alpha collected Nova's billing requirements.",
    "2024-01-09 14:00 UTC: Session beta detected a payment webhook failure for Nova.",
    "2024-01-10 10:15 UTC: Session beta retried the webhook and confirmed recovery.",
    "2024-01-12 16:45 UTC: Session alpha scheduled a follow-up review with Nova.",
]

QUERIES = [
    (
        "agent_alpha",
        "What happened with Nova before 2024-01-09?",
    ),
    (
        "agent_beta",
        "What payment-related events happened between 2024-01-09 and 2024-01-10?",
    ),
    (
        "agent_alpha",
        "What was the latest follow-up after the recovery?",
    ),
]


def print_results(query: str, results: list[Any]) -> None:
    print(f"\nQuery: {query}")

    if not results:
        print("No temporal results found.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}:")
        pprint(getattr(result, "search_result", result))


async def main() -> None:
    await cognee.forget(everything=True)

    await cognee.add(EVENTS, dataset_name=DATASET_NAME)
    await cognee.cognify(datasets=[DATASET_NAME], temporal_cognify=True)

    for session_id, query in QUERIES:
        results = await cognee.search(
            query_text=query,
            query_type=SearchType.TEMPORAL,
            datasets=[DATASET_NAME],
            session_id=session_id,
            top_k=10,
        )
        print(f"\nSession: {session_id}")
        print_results(query, results)


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
