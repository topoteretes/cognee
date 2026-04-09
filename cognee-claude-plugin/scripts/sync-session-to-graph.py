#!/usr/bin/env python3
"""Bridge session cache entries into the permanent knowledge graph on session end.

Calls cognee.improve(session_ids=[...]) to run:
  1. Apply feedback weights from session scores
  2. Persist session Q&A into the permanent graph
  3. Default enrichment (triplet embeddings)
  4. Sync graph knowledge back into session cache

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

    result = await cognee.improve(
        dataset=DATASET,
        session_ids=[SESSION_ID],
        run_in_background=False,
    )

    # Log summary to stderr (visible in hook output, not in Claude's context)
    if result and isinstance(result, dict):
        for ds_id, run_info in result.items():
            status = getattr(run_info, "status", "unknown")
            print(f"cognee-sync: dataset={ds_id} status={status}", file=sys.stderr)
    else:
        print(f"cognee-sync: dataset={DATASET} session={SESSION_ID} completed", file=sys.stderr)


def main():
    # Read stdin (SessionEnd payload) but we only use env vars for config
    sys.stdin.read()

    try:
        asyncio.run(_sync())
    except Exception as exc:
        # Non-fatal: session sync failure should not crash Claude Code
        print(f"cognee-sync: failed ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
