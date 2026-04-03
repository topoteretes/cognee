#!/usr/bin/env python3
"""Search session memory for context relevant to the user's prompt.

Reads the UserPromptSubmit hook payload from stdin, searches the session
cache for matching entries, and returns them as additionalContext so
Claude has relevant session history automatically.

Environment:
    COGNEE_PLUGIN_DATASET  - dataset name (default: claude_sessions)
    COGNEE_SESSION_ID      - session id  (default: claude_code_session)
"""

import asyncio
import json
import os
import sys

SESSION_ID = os.environ.get("COGNEE_SESSION_ID", "claude_code_session")
TOP_K = 3


async def _search(query: str) -> list:
    from cognee.api.v2.recall.recall import _search_session
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    return await _search_session(
        query_text=query,
        session_id=SESSION_ID,
        top_k=TOP_K,
        user=user,
    )


def main():
    payload_raw = sys.stdin.read()
    if not payload_raw.strip():
        return

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return

    prompt = payload.get("prompt", "")
    if not prompt or len(prompt) < 5:
        return

    try:
        results = asyncio.run(_search(prompt))
    except Exception:
        return

    if not results:
        return

    # Build context from matching session entries
    lines = ["Relevant context from this session's memory:\n"]
    for entry in results:
        q = entry.get("question", "")
        a = entry.get("answer", "")
        t = entry.get("time", "")
        if q:
            lines.append(f"[{t}] Q: {q}")
        if a:
            # Truncate long answers to keep context manageable
            a_short = a[:500] + "..." if len(a) > 500 else a
            lines.append(f"A: {a_short}")
        lines.append("")

    context = "\n".join(lines).strip()
    if not context or context == "Relevant context from this session's memory:":
        return

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
