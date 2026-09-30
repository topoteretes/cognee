"""Self-hosted AI companion: a chat companion that remembers your life, running only on your machine.

Your journal notes are remembered into a small life graph (models.py) by a local model on
Ollama, stored in embedded databases (Kuzu, LanceDB, SQLite) inside this folder. Every chat
opens with a check-in built from memory: your active goals and the dates coming up. When a
chat ends, cognee improves memory from it, so the next chat knows what you told this one.

Nothing leaves the machine: the LLM and the embeddings run on Ollama, the databases are local
files, and telemetry is off. The provider is fixed to Ollama here, so an LLM key in .env is
never used by this cookbook.

Commands:
  ingest   Build the memory from scratch from a notes folder.
  sync     Remember notes that are new or changed since the last sync; --watch repeats it.
  serve    Run the API server and the UI, and keep syncing the notes folder in the background.
  chat     Chat with the companion. --message sends one message and exits.

ingest, sync and serve take --sample to use sample_data/notes instead of --notes.

Requires: `ollama serve` with the two models pulled (see README.md). The UI needs Node.js and
npm (or Docker).
Run: uv run python examples/cookbooks/self_hosted_companion/companion.py ingest --sample
     uv run python examples/cookbooks/self_hosted_companion/companion.py chat
     uv run python examples/cookbooks/self_hosted_companion/companion.py serve --sample
"""

import argparse
import asyncio
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

# Fully local: these settings are forced, not defaults, so nothing in .env can point the
# companion at a hosted model. Only the model names and the Ollama address are yours to pick.
OLLAMA = os.environ.get("COMPANION_OLLAMA_URL", "http://localhost:11434").rstrip("/")
os.environ.update(
    {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": os.environ.get("COMPANION_LLM_MODEL", "llama3.1:8b"),
        "LLM_ENDPOINT": OLLAMA,
        "LLM_API_KEY": "ollama",
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": os.environ.get("COMPANION_EMBEDDING_MODEL", "nomic-embed-text"),
        "EMBEDDING_ENDPOINT": f"{OLLAMA}/api/embed",
        "EMBEDDING_DIMENSIONS": os.environ.get("COMPANION_EMBEDDING_DIMENSIONS", "768"),
        "HUGGINGFACE_TOKENIZER": os.environ.get(
            "COMPANION_TOKENIZER", "nomic-ai/nomic-embed-text-v1.5"
        ),
        "CACHING": "true",
        "CACHE_BACKEND": "sqlite",
        "TELEMETRY_DISABLED": "1",
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    }
)
# Auto-feedback analyzes every chat turn with a second LLM call. On a local model that call
# takes as long as the answer itself, so it is off unless you turn it on.
os.environ.setdefault("AUTO_FEEDBACK", "false")

import cognee  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402
from cognee.shared.logging_utils import ERROR, setup_logging  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from models import EXTRACTION_PROMPT, LifeGraph  # noqa: E402

DATASET = "companion"
USER_NAME = os.environ.get("COMPANION_USER_NAME", "Jordan")
SAMPLE_NOTES = HERE / "sample_data" / "notes"
SAMPLE_INCOMING = HERE / "sample_data" / "incoming" / "notes"
STATE_FILE = HERE / ".state.json"

# Everything the companion knows lives in this folder: delete it and the companion forgets.
STORE = HERE / ".cognee_system"
cognee.config.system_root_directory(str(STORE))
cognee.config.data_root_directory(str(STORE / "data"))
cognee.config.set_graph_database_provider("kuzu")
cognee.config.set_vector_db_provider("lancedb")


# ---------------------------------------------------------------- 1. ingestion


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"notes": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def remember_notes(paths: list[Path], state: dict) -> None:
    await cognee.remember(
        [str(path) for path in paths],
        dataset_name=DATASET,
        graph_model=LifeGraph,
        custom_prompt=EXTRACTION_PROMPT,
        self_improvement=False,
    )
    for path in paths:
        state["notes"][str(path)] = path.stat().st_mtime


def changed_notes(folder: Path, state: dict) -> list[Path]:
    """Markdown and text notes that are new, or edited since they were last remembered."""
    notes = sorted(p for p in folder.rglob("*") if p.suffix in {".md", ".txt"} and p.is_file())
    return [p for p in notes if state["notes"].get(str(p)) != p.stat().st_mtime]


async def ingest(notes_folder: Path) -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    state = {"notes": {}, "notes_folder": str(notes_folder)}
    notes = changed_notes(notes_folder, state)
    print(f"Remembering {len(notes)} notes from {notes_folder} with {os.environ['LLM_MODEL']}...")
    await remember_notes(notes, state)
    save_state(state)
    print("Memory is ready. Try: chat")


# ---------------------------------------------------------------- 2. keeping memory live


async def sync_once(sample: bool) -> str:
    state = load_state()
    folders = [Path(state.get("notes_folder", SAMPLE_NOTES))]
    if sample:
        # The sample stands in for a journal you keep writing: its incoming/ folder is what
        # "arrives" after the first ingest.
        folders.append(SAMPLE_INCOMING)
    notes = [note for folder in folders for note in changed_notes(folder, state)]
    if notes:
        await remember_notes(notes, state)
        save_state(state)
    return f"sync: {len(notes)} new or changed notes"


async def sync_forever(sample: bool, interval: int) -> None:
    while True:
        try:
            print(await sync_once(sample), flush=True)
        except Exception as error:  # noqa: BLE001 - a failed sync must not stop the next one.
            print(f"sync failed: {error}", flush=True)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------- 3. API server and UI


async def serve(sample: bool, interval: int, api_port: int, ui_port: int) -> None:
    """Run the API server, the UI and the notes sync in this one process.

    The graph database is embedded and only one process may open it, so the API server runs
    in-process (uvicorn) instead of as a child process. Syncs and UI requests then share it.
    """
    import uvicorn

    from cognee.api.client import app
    from cognee.api.v1.ui.ui import remove_ui_container, stop_ui_pid

    for port in (api_port, ui_port):
        with socket.socket() as probe:
            if probe.connect_ex(("localhost", port)) == 0:
                sys.exit(f"Port {port} is already in use: pass --api-port / --ui-port.")

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=api_port, log_level="warning")
    )
    api = asyncio.create_task(server.serve())
    while not server.started:
        if api.done():
            return await api  # Surfaces a startup error.
        await asyncio.sleep(0.2)

    # The UI assumes the API is on port 8000 unless COGNEE_BACKEND_URL says otherwise.
    os.environ["COGNEE_BACKEND_URL"] = f"http://localhost:{api_port}"
    spawned: list = []
    ui = await asyncio.to_thread(
        cognee.start_ui, pid_callback=spawned.append, port=ui_port, auto_download=True
    )
    print(
        f"\nUI:  http://localhost:{ui_port}"
        if ui
        else "\nThe UI did not start; the API still runs."
    )
    print(
        f"API: http://localhost:{api_port}\nSyncing notes every {interval}s. Press Ctrl+C to stop.\n"
    )

    syncing = asyncio.create_task(sync_forever(sample, interval))
    try:
        await api  # Returns when uvicorn handles Ctrl+C.
    finally:
        syncing.cancel()
        for item in spawned:
            pid, container = item if isinstance(item, tuple) else (item, None)
            if container:
                remove_ui_container(container)
            stop_ui_pid(pid)


# ---------------------------------------------------------------- 4. the companion


COMPANION_PROMPT = f"""
You are {USER_NAME}'s companion: warm, brief and practical. You know {USER_NAME} from their
journal and from earlier chats, which are in the context. Answer in two to four sentences.
Use what you remember when it helps, and say so when you don't know something. When
{USER_NAME} tells you something new, acknowledge it; you will remember it.
"""

CHECK_IN = (
    f"Today is {datetime.now().astimezone():%A, %B %-d, %Y}. Greet {USER_NAME} with a short check-in: "
    "mention one active goal and how it is going, and any date coming up in the next two weeks."
)


async def reply(message: str, session_id: str) -> str:
    # The session keeps the conversation: each answer sees the turns before it.
    results = await cognee.recall(
        message,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        session_id=session_id,
        system_prompt=COMPANION_PROMPT,
    )
    return results[0].text if results else "I don't have anything on that yet."


async def end_chat(session_id: str) -> None:
    # Improve turns the chat into long-term memory: its questions and answers are written into
    # the graph, and the preferences stated in it are updated.
    print("(saving this chat to memory...)", flush=True)
    await cognee.improve(dataset=DATASET, session_ids=[session_id])


async def chat(session_id: str | None, message: str | None, check_in: bool) -> None:
    session_id = session_id or f"chat-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    if message:
        print(f"companion> {await reply(message, session_id)}")
        return await end_chat(session_id)

    if check_in:
        print(f"companion> {await reply(CHECK_IN, session_id)}")
    print("(type /bye to end the chat)")
    while True:
        try:
            text = (await asyncio.to_thread(input, "you> ")).strip()
        except EOFError:
            break
        if text.lower() in {"/bye", "/exit", "/quit"}:
            break
        if text:
            print(f"companion> {await reply(text, session_id)}")
    await end_chat(session_id)


# ---------------------------------------------------------------- command line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "sync", "serve", "chat"):
        command = commands.add_parser(name)
        if name != "chat":
            command.add_argument("--sample", action="store_true", help="use sample_data/notes")
        if name == "ingest":
            command.add_argument("--notes", type=Path, help="your notes folder (.md, .txt)")
        if name in ("sync", "serve"):
            command.add_argument("--interval", type=int, default=60, help="seconds between syncs")
        if name == "sync":
            command.add_argument(
                "--watch", action="store_true", help="keep syncing every --interval"
            )
        if name == "serve":
            command.add_argument("--api-port", type=int, default=8000)
            command.add_argument("--ui-port", type=int, default=3000)
        if name == "chat":
            command.add_argument("--session", help="continue a session by its id")
            command.add_argument("--message", help="send one message and exit")
            command.add_argument(
                "--no-check-in", action="store_true", help="skip the opening check-in"
            )
    args = parser.parse_args()

    setup_logging(log_level=ERROR)
    if args.command == "ingest":
        if not args.sample and not args.notes:
            sys.exit("ingest needs --notes <folder> (or --sample).")
        asyncio.run(ingest(SAMPLE_NOTES if args.sample else args.notes.expanduser().resolve()))
    elif args.command == "sync":
        if args.watch:
            asyncio.run(sync_forever(args.sample, args.interval))
        else:
            print(asyncio.run(sync_once(args.sample)))
    elif args.command == "serve":
        asyncio.run(serve(args.sample, args.interval, args.api_port, args.ui_port))
    else:
        asyncio.run(chat(args.session, args.message, not args.no_check_in))


if __name__ == "__main__":
    main()
