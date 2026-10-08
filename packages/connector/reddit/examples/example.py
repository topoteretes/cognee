"""Reddit connector demo — give cognee a memory of a community.

``reddit_source`` returns a ``dlt`` source you hand to ``cognee.remember``.
Subreddit submissions (with a bounded comment tree per post) are ingested as
normal documents, so cognee extracts entities/tags from the conversation.

Each run is a full snapshot: unchanged posts keep a stable id and are not
re-cognified, and posts deleted from the subreddit drop out of the snapshot, so
cognee's orphan cleanup forgets them on the next sync.

────────────────────────────────────────────────────────────────────────────
Limits & privacy
────────────────────────────────────────────────────────────────────────────
The connector is read-only. It never posts, votes, or messages. Comment trees
are truncated (depth and comment-count ceilings) to keep documents reasonable,
with explicit truncation markers. User-facing fields are allow-listed; no
metadata beyond title/author/score/date/comments is written into documents.
"""

import asyncio
import os

import cognee

from cognee_community_connector_reddit import reddit_source

DATASET_NAME = "reddit"


async def main() -> None:
    missing = [
        var
        for var in (
            "REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET",
            "REDDIT_USERNAME",
            "REDDIT_PASSWORD",
        )
        if not os.environ.get(var)
    ]
    if missing:
        print(
            "Set these to run this example: "
            + ", ".join(missing)
            + "\n(create a Reddit 'script' app at https://www.reddit.com/prefs/apps)"
        )
        return

    source = reddit_source(subreddits=["machinelearning"], sort="hot", limit_per_subreddit=25)

    print("Syncing r/machinelearning into cognee ...")
    await cognee.remember(source, dataset_name=DATASET_NAME)

    answer = await cognee.search(
        query_text="What topics are the community excited about this week?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print("\nSearch result:\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
