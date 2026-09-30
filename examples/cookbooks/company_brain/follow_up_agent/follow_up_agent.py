"""Company brain follow-up agent: turn every call into confirmed Linear issues.

Granola calls, Gmail and Linear issues are remembered into one company graph (models.py).
After each new call the agent asks memory who the attendees are, which team and project each
next step belongs to, which deadlines apply and which steps Linear already tracks. It posts
the next steps it found to Slack and asks whether they are right. When someone confirms in
the thread, it creates one Linear issue per new step and remembers that they were created.

Commands:
  ingest     Build the memory from scratch. Calls already in it count as followed up.
  sync       Load only what is new since the last sync; --watch repeats it.
  follow-up  Sync, propose next steps for each new call, and act on answers from Slack.
  serve      Run the API server and the UI, and run follow-up in the background.

Every command takes --sample to use the files in sample_data/ instead of your accounts.
--dry-run prints the Slack message and the Linear issues instead of sending them.

Requires: LLM_API_KEY. Real data needs GRANOLA_API_KEY, LINEAR_API_KEY and/or cognee[gmail]
with a Gmail OAuth client; the agent needs SLACK_BOT_TOKEN and SLACK_CHANNEL (see README.md).
Run: uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py ingest --sample
     uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py follow-up --sample --dry-run
     uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py serve --sample
"""

import argparse
import asyncio
import os
import socket
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Local single-user mode: the SDK and the API server share one set of databases and the API
# needs no login. Load .env first so an explicit setting there still wins.
load_dotenv(override=False)
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

import cognee  # noqa: E402
from cognee.infrastructure.llm.LLMGateway import LLMGateway  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402
from cognee.shared.logging_utils import ERROR, setup_logging  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import actions  # noqa: E402
import sources  # noqa: E402
from models import EXTRACTION_PROMPT, CompanyGraph  # noqa: E402

DATASET = "company_brain_follow_up"
CALLS, EMAIL, LINEAR_ISSUES = "granola_calls", "email", "linear_issues"

# The cookbook keeps its memory in its own folder, so `ingest` can rebuild it from scratch
# without touching anything else cognee has stored on this machine.
STORE = HERE / ".cognee_system"
cognee.config.system_root_directory(str(STORE))
cognee.config.data_root_directory(str(STORE / "data"))


# ---------------------------------------------------------------- 1. ingestion


async def remember(data, node_set: str, **kwargs) -> None:
    """Remember one source into its own node set, extracted with the company graph model."""
    await cognee.remember(
        data,
        dataset_name=DATASET,
        node_set=[node_set],
        graph_model=CompanyGraph,
        custom_prompt=EXTRACTION_PROMPT,
        self_improvement=False,
        **kwargs,
    )


async def remember_batch(batch: dict[str, list[dict]], state: dict) -> None:
    if batch["linear_issues"]:
        await remember([sources.render_issue(i) for i in batch["linear_issues"]], LINEAR_ISSUES)
    if batch["emails"]:
        await remember([sources.render_email(e) for e in batch["emails"]], EMAIL)
    if batch["granola_notes"]:
        await remember([sources.render_note(n) for n in batch["granola_notes"]], CALLS)
        # A remembered call waits here until the agent has followed up on it.
        state.setdefault("calls_to_follow_up", {})
        for note in batch["granola_notes"]:
            state["calls_to_follow_up"][note["id"]] = note


async def ingest(sample: bool) -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    sources.reset_state()
    state = sources.load_state()

    if sample:
        batch = sources.sample_batch(sources.SAMPLE_DATA)
        print(
            f"Remembering {len(batch['granola_notes'])} calls, {len(batch['emails'])} emails "
            f"and {len(batch['linear_issues'])} Linear issues from sample_data/..."
        )
        await remember_batch(batch, state)
    else:
        print(await sync_once(sample=False, state=state))

    # Calls from before the agent started are history: it follows up only on new ones.
    state["followed_up_calls"] += list(state.pop("calls_to_follow_up", {}))
    sources.save_state(state)
    print("Memory is ready. Try: follow-up --sample --dry-run" if sample else "Memory is ready.")


# ---------------------------------------------------------------- 2. keeping memory live


async def sync_once(sample: bool, state: dict) -> str:
    """Remember only what arrived since the last sync, and report what that was."""
    if sample:
        # The sample stands in for live accounts: its incoming/ folder is what "arrives"
        # after the first ingest, and it arrives once.
        if state.get("sample_incoming_synced"):
            return "sync: nothing new"
        batch = sources.sample_batch(sources.SAMPLE_DATA / "incoming")
        await remember_batch(batch, state)
        state["sample_incoming_synced"] = True
        return f"sync: {len(batch['granola_notes'])} new calls"

    report = []
    batch = {"granola_notes": [], "emails": [], "linear_issues": []}
    if os.environ.get("GRANOLA_API_KEY"):
        batch["granola_notes"] = await sources.fetch_granola_notes(
            sources.since(state, "granola_since")
        )
        sources.advance(state, "granola_since", [n["created_at"] for n in batch["granola_notes"]])
        report.append(f"{len(batch['granola_notes'])} new calls")
    if os.environ.get("LINEAR_API_KEY"):
        batch["linear_issues"] = await sources.fetch_linear_issues(
            sources.since(state, "linear_since")
        )
        sources.advance(state, "linear_since", [i["updatedAt"] for i in batch["linear_issues"]])
        report.append(f"{len(batch['linear_issues'])} changed Linear issues")
    await remember_batch(batch, state)
    if Path(sources.GMAIL_CREDENTIALS).exists():
        # The Gmail connector remembers where it stopped: only new, changed and deleted
        # messages are loaded after the first run.
        inbox = sources.gmail_inbox()
        await remember(inbox, EMAIL, **sources.GMAIL_REMEMBER_KWARGS)
        report.append(f"Gmail: {inbox.cognee_sync_stats}")
    if not report:
        return "sync: no source configured (set GRANOLA_API_KEY, LINEAR_API_KEY or add Gmail credentials.json)"
    return "sync: " + ", ".join(report)


# ---------------------------------------------------------------- 3. the follow-up agent


class NextStep(BaseModel):
    title: str
    description: str
    owner: str | None = None
    team: str | None = None
    due_date: str | None = None
    existing_issue: str | None = None


class NextSteps(BaseModel):
    next_steps: list[NextStep]


NEXT_STEPS_PROMPT = """
You turn a call into Linear issues. List the concrete next steps agreed in the call: one per
action a person agreed to take.

For each next step give:
- title: a short imperative Linear issue title.
- description: one or two sentences with what to do and why.
- owner: the full name of the person who agreed to do it.
- team: the Linear team of the owner, from "Who the attendees are".
- due_date: YYYY-MM-DD when the call gives a date, or when a related email gives a deadline
  the step depends on; otherwise null.
- existing_issue: the identifier (such as "PAY-104") of a related Linear issue that already
  tracks this step; otherwise null.

Use only facts from the call and from memory. Never invent a date, a team or an issue.
"""


async def extract_next_steps(note: dict) -> list[dict]:
    """Research the call in memory, then turn the call and what was found into next steps."""
    call = sources.render_note(note)
    attendees = ", ".join(p.get("name", "") for p in note.get("attendees", []))

    # Who owns what is spread over many sources, so the graph answers it.
    teams = await cognee.recall(
        f"For each of these people, give their team, their role and the projects they work "
        f"on: {attendees}.",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        session_id=f"follow-up-{note['id']}",
    )

    # Issue identifiers and deadlines must be copied exactly, so the agent reads the source
    # text itself: the chunks of the Linear and email node sets closest to the call.
    async def closest(node_set: str, top_k: int) -> str:
        chunks = await cognee.recall(
            call,
            query_type=SearchType.CHUNKS,
            datasets=[DATASET],
            node_name=[node_set],
            top_k=top_k,
        )
        return "\n---\n".join(str(chunk.text) for chunk in chunks) or "(none)"

    found = await LLMGateway.acreate_structured_output(
        f"The call:\n{call}\n\n"
        f"Who the attendees are:\n{teams[0].text if teams else '(unknown)'}\n\n"
        f"Related Linear issues:\n{await closest(LINEAR_ISSUES, 5)}\n\n"
        f"Related emails:\n{await closest(EMAIL, 3)}",
        NEXT_STEPS_PROMPT,
        NextSteps,
    )
    return [step.model_dump() for step in found.next_steps]


async def act_on_answer(pending: dict, picked: list[int], dry_run: bool) -> None:
    """Create Linear issues for the confirmed steps, reply in Slack, and remember the result."""
    if not picked:
        await actions.reply_in_thread(pending, "Okay, I won't create any issues.", dry_run)
        return

    chosen = [s for n, s in enumerate(pending["steps"], start=1) if n in picked]
    new_steps = [s for s in chosen if not s.get("existing_issue")]
    tracked = [s for s in chosen if s.get("existing_issue")]
    created = await actions.create_linear_issues(new_steps, pending["call_title"], dry_run)

    reply = ["Created in Linear:", *(f"• {line}" for line in created)] if created else []
    reply += [f"Already tracked: {s['existing_issue']} ({s['title']})" for s in tracked]
    await actions.reply_in_thread(pending, "\n".join(reply) or "Nothing new to create.", dry_run)

    # Memory learns which steps are now tracked, so the next call does not propose them again.
    if created and not dry_run:
        await remember(
            f"Linear issues created after the call {pending['call_title']}:\n" + "\n".join(created),
            LINEAR_ISSUES,
        )


async def follow_up_once(sample: bool, dry_run: bool) -> None:
    state = sources.load_state()
    print(await sync_once(sample, state), flush=True)

    sources.save_state(state)

    for call_id, note in list(state.get("calls_to_follow_up", {}).items()):
        if call_id in state["followed_up_calls"]:  # Synced again, but already handled.
            del state["calls_to_follow_up"][call_id]
            continue
        print(f"Following up on the call {note['title']!r}...", flush=True)
        steps = await extract_next_steps(note)
        pending = {"call_title": note["title"], "steps": steps}
        if not steps:
            print("No next steps found.")
        elif dry_run:
            # Nobody can answer a message that was never sent: act as if all were confirmed.
            # The call stays in the queue, so a real run still follows up on it.
            await actions.ask_in_slack(note["title"], steps, dry_run)
            print("[dry run] Acting as if every step was confirmed.\n")
            await act_on_answer(pending, list(range(1, len(steps) + 1)), dry_run)
            continue
        else:
            pending |= await actions.ask_in_slack(note["title"], steps, dry_run)
            state["pending"][call_id] = pending
        state["followed_up_calls"].append(call_id)
        del state["calls_to_follow_up"][call_id]
        sources.save_state(state)

    sources.save_state(state)
    if dry_run:
        return

    # Answers can arrive long after the question: check every proposal still waiting.
    for call_id, pending in list(state["pending"].items()):
        picked = await actions.read_answer(pending)
        if picked is not None:
            print(
                f"Slack answered for {pending['call_title']!r}: steps {picked or 'none'}",
                flush=True,
            )
            await act_on_answer(pending, picked, dry_run)
            del state["pending"][call_id]
            sources.save_state(state)
    if state["pending"]:
        print(f"Waiting for Slack answers on {len(state['pending'])} calls.", flush=True)


async def repeat(job, interval: int) -> None:
    while True:
        try:
            await job()
        except Exception as error:  # noqa: BLE001 - one failed run must not stop the next one.
            print(f"run failed: {error}", flush=True)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------- 4. API server and UI


async def serve(sample: bool, dry_run: bool, interval: int, api_port: int, ui_port: int) -> None:
    """Run the API server, the UI and the follow-up loop in this one process.

    The graph database is embedded and only one process may open it, so the API server runs
    in-process (uvicorn) instead of as a child process. The agent and the UI then share it.
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
        f"API: http://localhost:{api_port}\nFollowing up every {interval}s. Press Ctrl+C to stop.\n"
    )

    agent = asyncio.create_task(repeat(lambda: follow_up_once(sample, dry_run), interval))
    try:
        await api  # Returns when uvicorn handles Ctrl+C.
    finally:
        agent.cancel()
        for item in spawned:
            pid, container = item if isinstance(item, tuple) else (item, None)
            if container:
                remove_ui_container(container)
            stop_ui_pid(pid)


# ---------------------------------------------------------------- command line


async def sync_and_save(sample: bool) -> None:
    state = sources.load_state()
    print(await sync_once(sample, state), flush=True)
    sources.save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "sync", "follow-up", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--sample", action="store_true", help="use sample_data/")
        if name in ("follow-up", "serve"):
            command.add_argument("--dry-run", action="store_true", help="print instead of sending")
        if name in ("sync", "follow-up", "serve"):
            command.add_argument("--interval", type=int, default=300, help="seconds between runs")
        if name in ("sync", "follow-up"):
            command.add_argument("--watch", action="store_true", help="repeat every --interval")
        if name == "serve":
            command.add_argument("--api-port", type=int, default=8000)
            command.add_argument("--ui-port", type=int, default=3000)
    args = parser.parse_args()

    setup_logging(log_level=ERROR)
    if args.command == "ingest":
        asyncio.run(ingest(args.sample))
    elif args.command == "sync":
        job = lambda: sync_and_save(args.sample)
        asyncio.run(repeat(job, args.interval) if args.watch else job())
    elif args.command == "follow-up":
        job = lambda: follow_up_once(args.sample, args.dry_run)
        asyncio.run(repeat(job, args.interval) if args.watch else job())
    else:
        asyncio.run(serve(args.sample, args.dry_run, args.interval, args.api_port, args.ui_port))


if __name__ == "__main__":
    main()
