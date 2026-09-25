"""Build a company brain from a relational database, a ticket export and meeting notes.

Three sources are ingested separately into one dataset, each into its own node set, and
extracted with one graph model (models.py). Nodes with the same identity merge, so the
person in the HR database, the assignee in the ticket export and the name in the meeting
notes become one node. The script then checks the result and starts the API server and
the UI, which Claude Code or Codex can connect to over MCP (see README.md).

Requires: LLM_API_KEY. The UI needs Node.js and npm (or Docker).
Run: uv run python examples/demos/company_brain/company_brain.py
     uv run python examples/demos/company_brain/company_brain.py --no-ui
     uv run python examples/demos/company_brain/company_brain.py --api-port 8010 --ui-port 3010
"""

import argparse
import asyncio
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from dlt.sources.sql_database import sql_database
from dotenv import load_dotenv

# Local single-user mode: the SDK, the API server and MCP share one set of databases
# and the API needs no login. Load .env first so an explicit setting there still wins.
load_dotenv(override=False)
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

import cognee  # noqa: E402
from cognee.infrastructure.databases.graph import get_graph_engine  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402
from cognee.shared.logging_utils import ERROR, setup_logging  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from models import EXTRACTION_PROMPT, CompanyGraph  # noqa: E402

DATASET = "company_brain"
DATA = Path(__file__).parent / "data"
DATABASE = DATA / "company.db"

HR_DATABASE = "hr_database"
SUPPORT_TICKETS = "support_tickets"
COMPANY_DOCS = "company_docs"

CROSS_SOURCE_QUESTION = (
    "Who is handling Brightline Retail's open high-priority ticket, which team are they "
    "on, and what fix was decided for it?"
)


def ensure_database() -> None:
    """Build company.db from schema.sql when it is missing."""
    if DATABASE.exists():
        return
    connection = sqlite3.connect(DATABASE)
    connection.executescript((DATA / "schema.sql").read_text())
    connection.close()


async def ingest_relational() -> None:
    """HR database: one text document per row of the three *_profiles views."""
    hr_database = sql_database(
        credentials=f"sqlite:///{DATABASE}",
        table_names=["employee_profiles", "project_profiles", "customer_profiles"],
        include_views=True,
    )
    # Opt into the dlt document path: each row becomes a document that is extracted with
    # the graph model. Without it, rows take the relational path, which builds its own
    # table/row graph and ignores graph_model, so nothing would link to other sources.
    hr_database.cognee_document_source = HR_DATABASE
    await cognee.remember(
        hr_database,
        dataset_name=DATASET,
        node_set=[HR_DATABASE],
        graph_model=CompanyGraph,
        custom_prompt=EXTRACTION_PROMPT,
    )


async def ingest_structured() -> None:
    """Support desk export: one JSON file of tickets."""
    await cognee.remember(
        str(DATA / "tickets.json"),
        dataset_name=DATASET,
        node_set=[SUPPORT_TICKETS],
        graph_model=CompanyGraph,
        custom_prompt=EXTRACTION_PROMPT,
    )


async def ingest_unstructured() -> None:
    """Meeting notes, a postmortem and a planning memo."""
    await cognee.remember(
        sorted(str(path) for path in (DATA / "docs").glob("*.md")),
        dataset_name=DATASET,
        node_set=[COMPANY_DOCS],
        graph_model=CompanyGraph,
        custom_prompt=EXTRACTION_PROMPT,
    )


MODEL_TYPES = {"Person", "Team", "Project", "Customer", "Ticket"}


async def verify() -> None:
    """Print what the graph holds and ask the questions that prove the sources linked."""
    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    by_id = {node_id: props for node_id, props in nodes}

    print("\n== Nodes per type (one per real entity) ==")
    counts = Counter(props.get("type") for props in by_id.values())
    for node_type in sorted(MODEL_TYPES):
        print(f"  {node_type:9} {counts.get(node_type, 0)}")

    # Dana Kim is in all three sources. One node that carries both her team and manager
    # (from the database) and her ticket (from the export) is what "the sources linked"
    # looks like.
    dana_ids = [
        i for i, p in by_id.items() if p.get("type") == "Person" and p.get("name") == "Dana Kim"
    ]
    print(f"\n== Dana Kim: {len(dana_ids)} Person node, connected to ==")
    for source_id, target_id, relationship, _ in edges:
        if dana_ids and dana_ids[0] in (source_id, target_id):
            other = by_id.get(target_id if source_id == dana_ids[0] else source_id, {})
            if other.get("type") in MODEL_TYPES:
                label = other.get("name") or other.get("ticket_id")
                print(f"  {relationship:12} {other.get('type'):8} {label}")

    # A node set tags the documents and chunks of one source, so a CHUNKS search with
    # node_name reads only that source's text.
    for node_set in (HR_DATABASE, SUPPORT_TICKETS, COMPANY_DOCS):
        print(f"\n== Chunks about Atlas from {node_set} only ==")
        results = await cognee.recall(
            "Who works on the Atlas project?",
            query_type=SearchType.CHUNKS,
            datasets=[DATASET],
            node_name=[node_set],
            top_k=3,
        )
        for result in results:
            print(f"  {' '.join(str(result.text).split())[:90]}")

    print(f"\n== Cross-source question ==\n  Q: {CROSS_SOURCE_QUESTION}")
    results = await cognee.recall(CROSS_SOURCE_QUESTION, datasets=[DATASET])
    print(f"  A: {results[0].text if results else '(no answer)'}")


def serve_ui(api_port: int, ui_port: int) -> None:
    """Start the API server and the UI until Ctrl+C."""
    from cognee.api.v1.ui.ui import remove_ui_container, stop_ui_pid

    spawned_pids: list[int] = []
    containers: list[str] = []

    def remember_process(pid_or_container):
        if isinstance(pid_or_container, tuple):
            pid, container = pid_or_container
            spawned_pids.append(pid)
            containers.append(container)
        else:
            spawned_pids.append(pid_or_container)

    def shut_down(*_):
        print("\nStopping the UI and API server...")
        for container in containers:
            remove_ui_container(container)
        for pid in spawned_pids:
            stop_ui_pid(pid)
        sys.exit(0)

    signal.signal(signal.SIGINT, shut_down)
    signal.signal(signal.SIGTERM, shut_down)

    # start_ui does not pass backend_port to the frontend, which assumes port 8000
    # unless COGNEE_BACKEND_URL says otherwise. The frontend inherits this environment.
    os.environ.setdefault("COGNEE_BACKEND_URL", f"http://localhost:{api_port}")

    ui_process = cognee.start_ui(
        pid_callback=remember_process,
        port=ui_port,
        auto_download=True,
        start_backend=True,
        backend_port=api_port,
    )
    if ui_process is None:
        sys.exit("The UI did not start; see the log above.")

    print(f"\nUI:  http://localhost:{ui_port}")
    print(f"API: http://localhost:{api_port}  (connect Claude Code or Codex to it, see README.md)")
    print("Press Ctrl+C to stop.")
    while ui_process.poll() is None:
        time.sleep(1)


async def build_brain() -> None:
    setup_logging(log_level=ERROR)
    ensure_database()

    # Start from an empty store, so the graph holds only this demo's data. This deletes
    # everything cognee has stored in its configured data and system directories.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print("Remembering the HR database...")
    await ingest_relational()
    print("Remembering the support tickets...")
    await ingest_structured()
    print("Remembering the company documents...")
    await ingest_unstructured()

    await verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-ui", action="store_true", help="skip the API server and UI")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    parser.add_argument("--ui-port", type=int, default=3000, help="UI port")
    arguments = parser.parse_args()

    # Check the ports before the two-minute ingestion, not after it.
    if not arguments.no_ui:
        for port in (arguments.api_port, arguments.ui_port):
            with socket.socket() as probe:
                if probe.connect_ex(("localhost", port)) == 0:
                    sys.exit(
                        f"Port {port} is already in use. Stop the process on it, or pass "
                        "--api-port / --ui-port (or --no-ui)."
                    )

    if arguments.no_ui:
        asyncio.run(build_brain())
    else:
        # The graph database is embedded and only one process may hold its file lock.
        # Build the brain in a child process, so the lock is released when it exits,
        # before the API server (another process) opens the same database.
        subprocess.run([sys.executable, __file__, "--no-ui"], check=True)
        serve_ui(arguments.api_port, arguments.ui_port)
