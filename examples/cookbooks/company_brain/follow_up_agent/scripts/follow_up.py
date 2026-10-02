"""Work out the next steps of your latest Granola call and post them to Slack.

Memory answers who owns each step, which team it belongs to, its deadline, and whether
Linear already tracks it: the team comes from earlier calls, a deadline from an email, a
tracked issue from Linear. Posts to Slack when SLACK_BOT_TOKEN and SLACK_CHANNEL are set
(a bot with chat:write, invited to the channel); prints the steps otherwise. With --sample,
it follows up the latest sample call and only prints, so sample data never reaches Slack.

Run alone: uv run python examples/cookbooks/company_brain/follow_up_agent/scripts/follow_up.py [--days N] [--sample]
"""

import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import httpx

import cognee  # also loads .env, so keys set there are seen
from cognee.modules.search.types import SearchType

DATASET = "company_brain"  # the same in every script
SAMPLE = Path(__file__).parent.parent / "sample"

NEXT_STEPS_PROMPT = """List the next steps agreed in the call, one per action a person agreed
to take. For each give: a short issue title, the owner, the owner's team, the due date, and
the Linear issue that already tracks it (such as PAY-104), if any. Use only facts from the
context: take teams from earlier calls and issues, and deadlines from emails. Never invent
a date, a team or an issue."""


def latest_call(days: int, sample: bool = False) -> tuple[str, str]:
    """The title and text of your newest Granola call from the last ``days`` days."""
    if sample:
        text = max((SAMPLE / "calls").glob("*.txt")).read_text()  # files are named in date order
        return text.splitlines()[0].removeprefix("Call: "), text
    if not os.environ.get("GRANOLA_API_KEY"):
        raise SystemExit("[follow_up] MISSING: GRANOLA_API_KEY is not set (put it in .env).")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {"Authorization": f"Bearer {os.environ['GRANOLA_API_KEY']}"}
    api = httpx.Client(base_url="https://public-api.granola.ai/v1/", headers=headers, timeout=60)
    # Read every page: the API returns at most 30 notes a page, in no promised order.
    notes, params = [], {"created_after": since, "page_size": 30}
    while True:
        page = api.get("notes", params=params).raise_for_status().json()
        notes.extend(page["notes"])
        if not page.get("hasMore"):
            break
        params["cursor"] = page["cursor"]
    if not notes:
        raise SystemExit(f"[follow_up] No Granola calls in the last {days} days.")
    newest = max(notes, key=lambda note: note["created_at"])
    note = api.get(f"notes/{newest['id']}", params={"include": "transcript"}).json()
    attendees = ", ".join(a.get("name") or a["email"] for a in note.get("attendees") or [])
    transcript = "\n".join(
        f"{(turn['speaker'] or {}).get('name') or 'Speaker'}: {turn['text']}"
        for turn in note.get("transcript") or []
    )
    text = (
        f"Call: {note.get('title')}\nDate: {note.get('created_at')}\n"
        f"Attendees: {attendees}\n\n{note.get('summary_text') or ''}\n\n{transcript}"
    )
    return note.get("title") or "your latest call", text


async def follow_up(days: int = 30, sample: bool = False) -> None:
    title, call = latest_call(days, sample)
    print(f"[follow_up] Call: {title}")
    # A graph search for the whole call finds the call and its issues, but an email that
    # sets a deadline rarely ranks, so fetch the emails about the call directly.
    emails = await cognee.recall(
        call, query_type=SearchType.CHUNKS, datasets=[DATASET], node_name=["email"], top_k=3
    )
    steps = await cognee.recall(
        # Ask in the question itself: an instruction only in the system prompt gets a
        # reply like "Got it, I've noted the call" instead of the steps.
        f"List the next steps agreed in this call, with owner, team, due date and the Linear "
        f"issue that tracks each one. When neither the call nor the issue gives a due date, "
        f"use a deadline from an email.\n\nThe call:\n{call}\n\nEmails about it:\n"
        + "\n---\n".join(str(email.text) for email in emails),
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        system_prompt=NEXT_STEPS_PROMPT,
    )
    if not steps:
        raise SystemExit("[follow_up] Nothing found. Run the ingest scripts first.")
    message = f'*Next steps from "{title}"*\n{steps[0].text}'

    if sample:
        print(f"[follow_up] Steps (a sample run, so not posted):\n{message}")
        return
    if not (os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_CHANNEL")):
        print(f"[follow_up] Steps (Slack not set up, so not posted):\n{message}")
        return
    async with httpx.AsyncClient() as http:
        posted = await http.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"},
            json={"channel": os.environ["SLACK_CHANNEL"], "text": message},
        )
    if not posted.json().get("ok"):  # Slack reports errors inside a 200 response
        raise SystemExit(f"[follow_up] Slack refused the message: {posted.json().get('error')}")
    print(f"[follow_up] Posted to Slack:\n{message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30, help="how far back to look for a call")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    args = parser.parse_args()
    asyncio.run(follow_up(args.days, args.sample))
