"""Company brain follow-up agent: turn your latest call into next steps, posted to Slack.

cognee remembers your Granola calls, Linear issues and Gmail inbox into one company graph.
Then it works out the next steps of your latest call: who owns each one, which team it
belongs to, its deadline, and whether Linear already tracks it. The steps are posted to
Slack, or printed when Slack is not set up.

Run: uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.ingestion.connectors import gmail_source

DATASET = "company_brain"
HERE = Path(__file__).parent
SINCE = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

NEXT_STEPS_PROMPT = """List the next steps agreed in the call, one per action a person agreed
to take. For each give: a short issue title, the owner, the owner's team, the due date, and
the Linear issue that already tracks it (such as PAY-104), if any. Use only facts from the
context: take teams from earlier calls and issues, and deadlines from emails. Never invent
a date, a team or an issue."""


def granola_calls() -> list[dict]:
    """Your Granola calls from the last 30 days, newest first, each with its text."""
    headers = {"Authorization": f"Bearer {os.environ['GRANOLA_API_KEY']}"}
    api = httpx.Client(base_url="https://public-api.granola.ai/v1/", headers=headers, timeout=60)
    calls, params = [], {"created_after": SINCE, "page_size": 30}
    while True:
        page = api.get("notes", params=params).raise_for_status().json()
        for listed in page["notes"]:
            note = api.get(f"notes/{listed['id']}", params={"include": "transcript"}).json()
            attendees = ", ".join(a.get("name") or a["email"] for a in note.get("attendees") or [])
            transcript = "\n".join(
                f"{(turn['speaker'] or {}).get('name') or 'Speaker'}: {turn['text']}"
                for turn in note.get("transcript") or []
            )
            text = (
                f"Call: {note.get('title')}\nDate: {note.get('created_at')}\n"
                f"Attendees: {attendees}\n\n{note.get('summary_text') or ''}\n\n{transcript}"
            )
            calls.append(
                {"title": note.get("title"), "created_at": note["created_at"], "text": text}
            )
        if not page.get("hasMore"):
            return sorted(calls, key=lambda call: call["created_at"], reverse=True)
        params["cursor"] = page["cursor"]


def linear_issues() -> list[str]:
    """Linear issues changed in the last 30 days, each as text."""
    query = """query($since: DateTimeOrDuration!) {
      issues(filter: {updatedAt: {gt: $since}}, first: 250) { nodes {
        identifier title description dueDate url
        state { name } assignee { name } team { name } project { name }
      } }
    }"""
    response = httpx.post(
        "https://api.linear.app/graphql",
        json={"query": query, "variables": {"since": SINCE}},
        headers={"Authorization": os.environ["LINEAR_API_KEY"]},  # a personal API key
        timeout=60,
    )
    issues = response.raise_for_status().json()["data"]["issues"]["nodes"]
    name = lambda field: (field or {}).get("name") or "none"
    return [
        f"Linear issue {i['identifier']}: {i['title']}\nStatus: {name(i['state'])}\n"
        f"Team: {name(i['team'])}\nProject: {name(i['project'])}\n"
        f"Assignee: {name(i['assignee'])}\nDue: {i['dueDate'] or 'none'}\n\n{i['description'] or ''}"
        for i in issues
    ]


async def main(ui: bool = False) -> None:
    # 1. Remember calls, issues and email, each in its own node set.
    print("Remembering your Granola calls...")
    calls = granola_calls()
    if not calls:
        sys.exit("No Granola calls in the last 30 days: nothing to follow up on.")
    await cognee.remember(
        [call["text"] for call in calls],
        dataset_name=DATASET,
        node_set=["calls"],
        self_improvement=False,
    )
    if os.environ.get("LINEAR_API_KEY"):
        print("Remembering your Linear issues...")
        await cognee.remember(
            linear_issues(), dataset_name=DATASET, node_set=["linear"], self_improvement=False
        )
    if (HERE / "credentials.json").exists():
        print("Remembering your Gmail inbox...")
        await cognee.remember(
            # cognee's Gmail connector; max_results keeps the first try small.
            gmail_source(
                credentials_path=str(HERE / "credentials.json"),
                token_path=str(HERE / "token.json"),
                label_ids=["INBOX"],
                max_results=50,
            ),
            dataset_name=DATASET,
            node_set=["email"],
            write_disposition="merge",  # the connector's rows are merged by message id
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )

    # 2. Ask memory for the next steps of the latest call.
    latest = calls[0]
    steps = await cognee.recall(
        f"The call:\n{latest['text']}",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        system_prompt=NEXT_STEPS_PROMPT,
    )
    message = f'*Next steps from "{latest["title"]}"*\n{steps[0].text}'

    # 3. Post them to Slack (a bot with chat:write, invited to the channel), or print them.
    if os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_CHANNEL"):
        async with httpx.AsyncClient() as http:
            posted = await http.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"},
                json={"channel": os.environ["SLACK_CHANNEL"], "text": message},
            )
        if not posted.json().get("ok"):  # Slack reports errors inside a 200 response
            sys.exit(f"Slack refused the message: {posted.json().get('error')}")
        print(f"Posted to Slack:\n{message}")
    else:
        print(message)

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
    if not os.environ.get("GRANOLA_API_KEY"):
        sys.exit("Set GRANOLA_API_KEY: the agent follows up on your Granola calls.")
    setup_logging(log_level=ERROR)
    asyncio.run(main("--ui" in sys.argv))
