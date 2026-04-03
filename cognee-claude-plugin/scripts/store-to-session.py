#!/usr/bin/env python3
"""Store text into a Cognee session cache (lightweight, no cognify).

Usage:
    echo '{"tool_name":"Read","tool_input":{},"tool_output":"..."}' | python store-to-session.py
    echo '{"assistant_message":"..."}' | python store-to-session.py --stop

Environment:
    COGNEE_PLUGIN_DATASET  - dataset name (default: claude_sessions)
    COGNEE_SESSION_ID      - session id  (default: claude_code_session)
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone


DATASET = os.environ.get("COGNEE_PLUGIN_DATASET", "claude_sessions")
SESSION_ID = os.environ.get("COGNEE_SESSION_ID", "claude_code_session")
MAX_TEXT = 4000


def _build_tool_text(payload: dict) -> str:
    tool_name = payload.get("tool_name", "unknown")
    tool_input = json.dumps(payload.get("tool_input", {}))[:MAX_TEXT]
    tool_response = str(payload.get("tool_response", ""))[:MAX_TEXT]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"[{ts}] Tool: {tool_name}\nInput: {tool_input}\nOutput: {tool_response}"


def _build_stop_text(payload: dict) -> str:
    msg = str(payload.get("last_assistant_message", ""))[:MAX_TEXT]
    if not msg or msg == "null":
        return ""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"[{ts}] Assistant response:\n{msg}"


async def _store(text: str):
    """Call cognee.remember with session_id for lightweight session storage."""
    import cognee

    result = await cognee.remember(
        data=text,
        dataset_name=DATASET,
        session_id=SESSION_ID,
    )

    if not result:
        print(
            f"cognee-session: store failed (status={result.status})",
            file=sys.stderr,
        )


def main():
    payload_raw = sys.stdin.read()
    if not payload_raw.strip():
        return

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return

    is_stop = "--stop" in sys.argv

    # Skip recursive cognee-cli / cognee calls
    if not is_stop:
        tool_name = payload.get("tool_name", "")
        tool_input_str = json.dumps(payload.get("tool_input", {}))
        if tool_name == "Bash" and "cognee" in tool_input_str:
            return
        text = _build_tool_text(payload)
    else:
        text = _build_stop_text(payload)

    if not text:
        return

    asyncio.run(_store(text))


if __name__ == "__main__":
    main()
