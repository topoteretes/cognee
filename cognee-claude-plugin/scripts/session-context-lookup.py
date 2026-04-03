#!/usr/bin/env python3
"""Search session memory for context relevant to the user's prompt.

Reads the UserPromptSubmit hook payload from stdin, searches the session
cache for matching entries, and returns them as additionalContext.

Uses cognee's cache engine directly (avoids heavy litellm/pipeline imports).
Total import cost: ~500ms (structlog + pydantic + sqlalchemy + diskcache).

Environment:
    COGNEE_SESSION_ID  - session id (default: claude_code_session)
"""

import asyncio
import json
import os
import re
import sys

SESSION_ID = os.environ.get("COGNEE_SESSION_ID", "claude_code_session")
TOP_K = 3
MIN_WORD_LEN = 2


def _tokenize(text: str) -> set:
    return {w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) >= MIN_WORD_LEN}


def _search_entries(entries: list, query_text: str) -> list:
    query_words = _tokenize(query_text)
    if not query_words:
        return []

    scored = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_text = " ".join(str(entry.get(f, "")) for f in ("question", "context", "answer"))
        entry_words = _tokenize(entry_text)
        hits = len(query_words & entry_words)
        if hits > 0:
            scored.append((hits, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:TOP_K]]


async def _get_entries(session_id: str) -> list:
    """Load session entries via cognee's cache engine (lightweight imports only)."""
    from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    user_id = str(user.id) if hasattr(user, "id") else ""
    if not user_id:
        return []

    cache_engine = get_cache_engine()
    if cache_engine is None:
        return []

    try:
        entries = await cache_engine.get_all_qa_entries(user_id, session_id)
        return list(entries) if entries else []
    except Exception:
        return []


async def _run(prompt: str):
    entries = await _get_entries(SESSION_ID)
    if not entries:
        return

    results = _search_entries(entries, prompt)
    if not results:
        return

    lines = ["Relevant context from this session's memory:\n"]
    for entry in results:
        q = entry.get("question", "")
        a = entry.get("answer", "")
        t = entry.get("time", "")
        if q:
            lines.append(f"[{t}] Q: {q}")
        if a:
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
        asyncio.run(_run(prompt))
    except Exception:
        pass


if __name__ == "__main__":
    main()
