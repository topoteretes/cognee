"""Self-hosted AI companion: a chat companion that knows your notes.

cognee remembers your notes folder (a journal, an Obsidian vault, any .md or .txt files) into
a knowledge graph, then you chat with it. When the chat ends, cognee writes it into memory,
so the next chat knows what you said. cognee's databases are local files; the LLM is the one
your .env configures.

Run: uv run python examples/cookbooks/self_hosted_companion/companion.py ~/Documents/journal
"""

import asyncio
import os
import sys
from datetime import datetime

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

PROMPT = """You are the user's companion: warm, brief and practical. You know the user from
their notes and earlier chats, which are in the context. Answer in two to four sentences, and
say so when you don't know something."""


async def main(notes_folder: str, ui: bool = False) -> None:
    print(f"Remembering the notes in {notes_folder}...")
    await cognee.remember(notes_folder, dataset_name="companion", self_improvement=False)

    session_id = f"chat-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    print("Chat with your companion. Type /bye to end.")
    while True:
        try:
            message = input("you> ").strip()
        except EOFError:  # Ctrl+D
            break
        if message == "/bye":
            break
        results = await cognee.recall(
            message,
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=["companion"],
            session_id=session_id,  # each answer sees the turns before it
            system_prompt=PROMPT,
        )
        print("companion>", results[0].text if results else "I don't know that yet.")

    print("Saving this chat to memory...")
    await cognee.improve(dataset="companion", session_ids=[session_id])

    if ui:
        await open_ui()


async def open_ui() -> None:
    """Browse the graph: cognee's API server runs in this process, next to the databases it
    already has open, and the UI runs as its own process. Ctrl+C stops both."""
    import uvicorn

    from cognee.api.client import app
    from cognee.api.v1.ui.ui import remove_ui_container, stop_ui_pid

    server = uvicorn.Server(uvicorn.Config(app, port=8000, log_level="warning"))
    api = asyncio.create_task(server.serve())
    while not server.started:
        if api.done():
            return await api  # raises the startup error, such as a port in use
        await asyncio.sleep(0.2)
    spawned: list = []  # a PID, or (PID, container) when the UI runs in Docker
    await asyncio.to_thread(cognee.start_ui, spawned.append, auto_download=True)
    print("Browse the graph at http://localhost:3000. Press Ctrl+C to stop.")
    try:
        await api  # returns once uvicorn has handled Ctrl+C
    finally:
        for item in spawned:
            pid, container = item if isinstance(item, tuple) else (item, None)
            if container:
                remove_ui_container(container)
            stop_ui_pid(pid)


if __name__ == "__main__":
    folders = [arg for arg in sys.argv[1:] if arg != "--ui"]
    if len(folders) != 1 or not os.path.isdir(folders[0]):
        sys.exit("Usage: companion.py <your notes folder> [--ui]")
    setup_logging(log_level=ERROR)
    asyncio.run(main(folders[0], "--ui" in sys.argv))
