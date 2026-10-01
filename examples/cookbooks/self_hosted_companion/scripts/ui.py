"""Browse the graph: cognee's API server and the UI at http://localhost:3000.

The API server runs in this process, next to the databases cognee has open, and the UI runs
as its own process. Ctrl+C stops both.

Run alone: uv run python examples/cookbooks/self_hosted_companion/scripts/ui.py
"""

import asyncio

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging


async def open_ui() -> None:
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
    print("[ui] Browse the graph at http://localhost:3000. Press Ctrl+C to stop.")
    try:
        await api  # returns once uvicorn has handled Ctrl+C
    finally:
        for item in spawned:
            pid, container = item if isinstance(item, tuple) else (item, None)
            if container:
                remove_ui_container(container)
            stop_ui_pid(pid)


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(open_ui())
