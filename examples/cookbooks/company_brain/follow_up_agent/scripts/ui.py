"""Browse the graph: cognee's API server and the UI at http://localhost:3000.

The API server runs in this process, next to the databases cognee has open, and the UI runs
as its own process. Ctrl+C stops both.

Run alone: uv run python examples/cookbooks/company_brain/follow_up_agent/scripts/ui.py
"""

import asyncio
import os

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee


async def open_ui() -> None:
    import uvicorn

    from cognee.api.client import app
    from cognee.api.v1.ui.ui import remove_ui_container, stop_ui_pid
    from cognee.base_config import get_base_config

    # The UI signs in as cognee's default user, which has no password unless
    # DEFAULT_USER_PASSWORD gives it one. Like `cognee-cli -ui`, give it the well-known
    # local password: the server listens on localhost only, so only this machine can use it.
    config = get_base_config()
    config.default_user_password = config.default_user_password or "default_password"

    server = uvicorn.Server(uvicorn.Config(app, port=8000, log_level="warning"))
    api = asyncio.create_task(server.serve())
    while not server.started:
        if api.done():
            return await api  # raises the startup error, such as a port in use
        await asyncio.sleep(0.2)
    spawned: list = []  # a PID, or (PID, container) when the UI runs in Docker
    ui = await asyncio.to_thread(cognee.start_ui, spawned.append, auto_download=True)
    if ui is None:  # cognee logged why, such as port 3000 being in use
        server.should_exit = True
        await api
        raise SystemExit("[ui] The UI did not start; see the error above.")
    print("[ui] Browse the graph at http://localhost:3000. Press Ctrl+C to stop.")
    try:
        await api  # returns once uvicorn has handled Ctrl+C
    except asyncio.CancelledError:  # Ctrl+C: stop quietly instead of with a traceback
        pass
    finally:
        for item in spawned:
            pid, container = item if isinstance(item, tuple) else (item, None)
            if container:
                remove_ui_container(container)
            stop_ui_pid(pid)


if __name__ == "__main__":
    asyncio.run(open_ui())
