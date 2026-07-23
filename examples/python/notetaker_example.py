import asyncio

import cognee
from cognee.shared.logging_utils import setup_logging, INFO
from cognee.api.v1.search import SearchType
from cognee.api.v1.notetaker import _RECALL_PROMPTS
from cognee.tasks.notetaker.normalize import normalize_transcript


async def recall(series_id: str, query: str, query_type: str) -> list:
    """Temporal recall scoped to a single meeting series with a focused prompt."""
    return await cognee.search(
        query_text=query,
        query_type=SearchType.TEMPORAL,
        datasets=[series_id],
        system_prompt_path=_RECALL_PROMPTS[query_type],
        include_references=True,
    )


async def main():
    series_id = "engineering_standups"

    # 1. Two occurrences of the same series, a week apart. Because the series is
    #    the dataset, both land in one graph and "what changed" can span them.
    occurrence_1 = normalize_transcript(
        turns=[
            ("Alice", "Let's release v1 today.", "2026-06-10 10:00"),
            ("Bob", "I'll handle the deployment.", "2026-06-10 10:05"),
        ],
        meeting_id="standup_2026_06_10",
        permalink="https://example.com/standup/1",
    )
    occurrence_2 = normalize_transcript(
        turns=[
            ("Alice", "v1 is out. Let's plan v2 for next week.", "2026-06-17 10:00"),
            ("Bob", "I'll draft the v2 rollout doc.", "2026-06-17 10:04"),
        ],
        meeting_id="standup_2026_06_17",
        permalink="https://example.com/standup/2",
    )

    print(f"Normalized occurrence 1:\n{occurrence_1}\n")

    # 2. Ingest both occurrences into the series dataset and build the temporal graph.
    await cognee.add(occurrence_1, dataset_name=series_id)
    await cognee.add(occurrence_2, dataset_name=series_id)

    print("Cognifying with temporal awareness...")
    await cognee.cognify(datasets=[series_id], temporal_cognify=True)
    print("Cognify complete!\n")

    # 3. Recall action items, decisions, and the temporal delta — all scoped to the series.
    print("Action items:")
    print(await recall(series_id, "What are the action items?", "action_items"), "\n")

    print("Decisions:")
    print(await recall(series_id, "What decisions were made?", "decisions"), "\n")

    print("What changed since last week:")
    print(await recall(series_id, "What changed since last week?", "temporal_delta"), "\n")

    # 4. Forget the whole series when you're done.
    await cognee.forget(dataset=series_id)
    print("Series forgotten.")


if __name__ == "__main__":
    setup_logging(log_level=INFO)
    asyncio.run(main())
