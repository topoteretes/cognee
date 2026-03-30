#!/usr/bin/env python3
"""Bridge session cache entries into the permanent knowledge graph on session end.

Calls cognee.improve(session_ids=[...]) to run:
  1. Apply feedback weights from session scores
  2. Persist session Q&A into the permanent graph
  3. Default enrichment (triplet embeddings)

Reads SessionEnd hook payload from stdin for session_id context.

Environment:
    COGNEE_PLUGIN_DATASET  - dataset name (default: claude_sessions)
    COGNEE_SESSION_ID      - session id  (default: claude_code_session)
"""

import asyncio
import os
import sys


DATASET = os.environ.get("COGNEE_PLUGIN_DATASET", "claude_sessions")
SESSION_ID = os.environ.get("COGNEE_SESSION_ID", "claude_code_session")


async def _sync():
    import cognee

    await cognee.improve(
        dataset=DATASET,
        session_ids=[SESSION_ID],
        run_in_background=False,
    )


def main():
    # Read stdin (SessionEnd payload) but we only use env vars for config
    sys.stdin.read()

    try:
        asyncio.run(_sync())
    except Exception:
        # Non-fatal: session sync failure should not crash Claude Code
        pass


if __name__ == "__main__":
    main()
