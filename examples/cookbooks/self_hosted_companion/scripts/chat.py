"""Chat with a companion that knows your notes and every earlier chat.

Each message is a recall in the chat's session, so answers see the turns before them. When
the chat ends, improve() writes it into memory, so the next chat knows what you said. With
--ask, it answers one message and saves it the same way, without waiting for typed input.

Run alone: uv run python examples/cookbooks/self_hosted_companion/scripts/chat.py [--ask "message"]
"""

import argparse
import asyncio
from datetime import datetime

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "companion"  # the same in every script

PROMPT = """You are the user's companion: warm, brief and practical. You know the user from
their notes and earlier chats, which are in the context. Answer in two to four sentences, and
say so when you don't know something."""


async def reply(message: str, session_id: str) -> str:
    results = await cognee.recall(
        message,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        session_id=session_id,  # each answer sees the turns before it
        system_prompt=PROMPT,
    )
    return str(results[0].text) if results else "I don't know that yet."


async def chat(ask: str | None = None) -> None:
    session_id = f"chat-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    if ask:
        print(f"[chat] you> {ask}\n[chat] companion> {await reply(ask, session_id)}")
    else:
        print("[chat] Chat with your companion. Type /bye to end.")
        while True:
            try:
                message = input("you> ").strip()
            except EOFError:  # Ctrl+D
                break
            if message == "/bye":
                break
            print("companion>", await reply(message, session_id))

    await cognee.improve(dataset=DATASET, session_ids=[session_id])
    print("[chat] Saved this chat to memory.")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ask", help="answer one message instead of an interactive chat")
    asyncio.run(chat(parser.parse_args().ask))
