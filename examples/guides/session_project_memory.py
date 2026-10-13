"""Read a finished session's distilled project guidance and structured summary.

Run after improve has processed a real session:
    uv run python examples/guides/session_project_memory.py DATASET SESSION_ID

Requires a release advertising session_summary_v1/context_usage_v1 in /health.
Default recall stays read-only. This example explicitly stamps returned guidance as used;
unused agent guidance expires after 20 newer sessions of the same user and dataset
(COGNEE_AGENT_CONTEXT_UNUSED_SESSIONS overrides 20). Improve publishes summaries into
session_summaries and returns only a successfully published snapshot with guidance.
No raw transcript fallback is used. Existing graph memories are never removed by expiry.
"""

import asyncio
import sys

import cognee


async def main(dataset: str, session_id: str):
    entries = await cognee.recall(
        "project decisions, conventions, pitfalls and open tasks",
        datasets=[dataset],
        session_id=session_id,
        scope=["session_context"],
        context_profile="agent",
        only_context=True,
        record_context_usage=True,
    )
    for entry in entries:
        print(entry.content)
        if entry.session_summary:
            print(entry.session_summary)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:3]))
