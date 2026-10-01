"""Work out the next steps of your latest Granola call and post them to Slack.

Memory answers who owns each step, which team it belongs to, its deadline, and whether
Linear already tracks it: the team comes from earlier calls, a deadline from an email, a
tracked issue from Linear. Posts to Slack when SLACK_BOT_TOKEN and SLACK_CHANNEL are set
(a bot with chat:write, invited to the channel); prints the steps otherwise.

Run alone: uv run python examples/cookbooks/company_brain/follow_up_agent/scripts/follow_up.py [--days N]
"""

import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone

import httpx

import cognee  # also loads .env, so keys set there are seen
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "company_brain"  # the same in every script

NEXT_STEPS_PROMPT = """List the next steps agreed in the call, one per action a person agreed
to take. For each give: a short issue title, the owner, the owner's team, the due date, and
the Linear issue that already tracks it (such as PAY-104), if any. Use only facts from the
context: take teams from earlier calls and issues, and deadlines from emails. Never invent
a date, a team or an issue."""


def latest_call(days: int) -> tuple[str, str]:
    """The title and text of your newest Granola call from the last ``days`` days."""
    if not os.environ.get("GRANOLA_API_KEY"):
        raise SystemExit("[follow_up] MISSING: GRANOLA_API_KEY is not set (put it in .env).")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {"Authorization": f"Bearer {os.environ['GRANOLA_API_KEY']}"}
    api = httpx.Client(base_url="https://public-api.granola.ai/v1/", headers=headers, timeout=60)
    listed = api.get("notes", params={"created_after": since, "page_size": 30})
    notes = listed.raise_for_status().json()["notes"]
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


async def follow_up(days: int = 30) -> None:
    title, call = latest_call(days)
    print(f"[follow_up] Call: {title}")
    steps = await cognee.recall(
        f"The call:\n{call}",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        system_prompt=NEXT_STEPS_PROMPT,
    )
    if not steps:
        raise SystemExit("[follow_up] Nothing found. Run the ingest scripts first.")
    message = f'*Next steps from "{title}"*\n{steps[0].text}'

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
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30, help="how far back to look for a call")
    asyncio.run(follow_up(parser.parse_args().days))
