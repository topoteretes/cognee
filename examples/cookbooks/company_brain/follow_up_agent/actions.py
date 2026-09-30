"""What the follow-up agent does in the outside world: ask in Slack, then file Linear issues.

Slack is asked with a plain message, and the answer is read back from its thread and its
reactions. That needs no public URL, unlike Slack's interactive buttons, so the agent runs
on a laptop. The bot token needs the ``chat:write``, ``channels:history`` (or
``groups:history`` for a private channel) and ``reactions:read`` scopes.

With ``dry_run`` nothing is sent: the Slack message and each Linear payload are printed.
"""

import os
import re

import aiohttp
from sources import linear_graphql

SLACK_API = "https://slack.com/api/"
YES_REACTIONS = {"white_check_mark", "heavy_check_mark", "+1", "thumbsup"}
NO_REACTIONS = {"x", "-1", "thumbsdown"}


# ---------------------------------------------------------------- Slack


async def slack(method: str, **payload) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"}
    async with aiohttp.ClientSession(headers=headers) as http:
        if method.startswith(("conversations.", "reactions.")):
            request = http.get(SLACK_API + method, params=payload)
        else:
            request = http.post(SLACK_API + method, json=payload)
        async with request as response:
            body = await response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {body.get('error')}")
    return body


def proposal_message(call_title: str, steps: list[dict]) -> str:
    lines = [f'*Follow-up for "{call_title}"*', "I think these are the next steps:"]
    for number, step in enumerate(steps, start=1):
        details = " · ".join(
            v for v in (step.get("owner"), step.get("team"), step.get("due_date")) if v
        )
        tracked = (
            f" — already tracked in {step['existing_issue']}" if step.get("existing_issue") else ""
        )
        lines.append(f"{number}. {step['title']}" + (f" ({details})" if details else "") + tracked)
    lines.append(
        "Reply *yes* in this thread, or *yes 1 3* for some of them, or react with :white_check_mark:,"
        " and I'll create the Linear issues. Reply *no* or react with :x: to skip."
    )
    return "\n".join(lines)


async def ask_in_slack(call_title: str, steps: list[dict], dry_run: bool) -> dict:
    """Post the proposed next steps and return where the answer will appear."""
    text = proposal_message(call_title, steps)
    if dry_run:
        print(
            f"\n[dry run] Slack message to {os.environ.get('SLACK_CHANNEL', '<SLACK_CHANNEL>')}:\n{text}\n"
        )
        return {"channel": None, "ts": None}
    posted = await slack("chat.postMessage", channel=os.environ["SLACK_CHANNEL"], text=text)
    return {"channel": posted["channel"], "ts": posted["ts"]}


def parse_answer(text: str, step_count: int) -> list[int] | None:
    """'yes' -> every step, 'yes 1 3' -> steps 1 and 3, 'no' -> [], anything else -> None."""
    words = text.strip().lower()
    if re.match(r"^(no|nope|skip)\b", words):
        return []
    if re.match(r"^(yes|yep|y|correct|lgtm)\b", words):
        picked = [int(n) for n in re.findall(r"\d+", words) if 1 <= int(n) <= step_count]
        return picked or list(range(1, step_count + 1))
    return None


async def read_answer(pending: dict) -> list[int] | None:
    """The steps a person confirmed in Slack, [] if they declined, None while nobody answered."""
    step_count = len(pending["steps"])
    replies = await slack("conversations.replies", channel=pending["channel"], ts=pending["ts"])
    for message in replies["messages"][1:]:  # [0] is the proposal itself.
        if message.get("bot_id"):
            continue
        answer = parse_answer(message.get("text", ""), step_count)
        if answer is not None:
            return answer

    reactions = await slack("reactions.get", channel=pending["channel"], timestamp=pending["ts"])
    names = {r["name"] for r in reactions["message"].get("reactions", [])}
    if names & YES_REACTIONS:
        return list(range(1, step_count + 1))
    if names & NO_REACTIONS:
        return []
    return None


async def reply_in_thread(pending: dict, text: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry run] Slack thread reply:\n{text}\n")
        return
    await slack("chat.postMessage", channel=pending["channel"], thread_ts=pending["ts"], text=text)


# ---------------------------------------------------------------- Linear


TEAMS_AND_USERS_QUERY = """
query { teams { nodes { id name key } } users { nodes { id name email } } }
"""

CREATE_ISSUE_MUTATION = """
mutation Create($input: IssueCreateInput!) {
  issueCreate(input: $input) { success issue { identifier url } }
}
"""


async def create_linear_issues(steps: list[dict], call_title: str, dry_run: bool) -> list[str]:
    """Create one Linear issue per confirmed step and return a line per created issue."""
    if dry_run:
        teams, users = {}, {}
    else:
        directory = await linear_graphql(TEAMS_AND_USERS_QUERY)
        teams = {t["name"].lower(): t["id"] for t in directory["teams"]["nodes"]}
        teams |= {t["key"].lower(): t["id"] for t in directory["teams"]["nodes"]}
        users = {u["name"].lower(): u["id"] for u in directory["users"]["nodes"]}

    default_team = os.environ.get("LINEAR_DEFAULT_TEAM", "")
    created = []
    for step in steps:
        team = step.get("team") or default_team
        issue = {
            "teamId": teams.get(team.lower(), teams.get(default_team.lower()))
            or f"<id of team {team}>",
            "title": step["title"],
            "description": f"{step['description']}\n\nFrom the call *{call_title}*, confirmed in Slack.",
        }
        if step.get("owner"):
            issue["assigneeId"] = users.get(step["owner"].lower()) or f"<id of {step['owner']}>"
        if step.get("due_date"):
            issue["dueDate"] = step["due_date"]

        if dry_run:
            print(f"[dry run] Linear issueCreate input: {issue}")
            created.append(f"(dry run) {step['title']}")
            continue
        if issue["teamId"].startswith("<"):
            created.append(
                f"skipped {step['title']!r}: no Linear team {team!r} (set LINEAR_DEFAULT_TEAM)"
            )
            continue
        if issue.get("assigneeId", "").startswith("<"):
            del issue["assigneeId"]  # Unknown in Linear: leave it unassigned, don't fail.
        result = await linear_graphql(CREATE_ISSUE_MUTATION, {"input": issue})
        made = result["issueCreate"]["issue"]
        created.append(f"{made['identifier']} {step['title']} {made['url']}")
    return created
