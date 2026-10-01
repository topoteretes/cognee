"""Remember the Linear issues changed in the last 30 days (node set `linear`).

Needs LINEAR_API_KEY, a personal API key (Linear: Settings → Security & access), in .env at
the repo root.

Run alone: uv run python examples/cookbooks/company_brain/follow_up_agent/scripts/ingest_linear.py [--days N]
"""

import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone

import httpx

import cognee  # also loads .env, so a LINEAR_API_KEY set there is seen
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "company_brain"  # the same in every script

QUERY = """query($since: DateTimeOrDuration!) {
  issues(filter: {updatedAt: {gt: $since}}, first: 250) { nodes {
    identifier title description dueDate url
    state { name } assignee { name } team { name } project { name }
  } }
}"""


def linear_issues(days: int) -> list[str]:
    """Linear issues changed in the last ``days`` days, each as text."""
    if not os.environ.get("LINEAR_API_KEY"):
        raise SystemExit("[ingest_linear] MISSING: LINEAR_API_KEY is not set (put it in .env).")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    response = httpx.post(
        "https://api.linear.app/graphql",
        json={"query": QUERY, "variables": {"since": since}},
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


async def ingest_linear(days: int = 30) -> None:
    issues = linear_issues(days)
    if not issues:
        print(f"[ingest_linear] No Linear issues changed in the last {days} days.")
        return
    await cognee.remember(issues, dataset_name=DATASET, node_set=["linear"], self_improvement=False)
    print(f"[ingest_linear] Remembered {len(issues)} Linear issues from the last {days} days")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30, help="how far back to read issues")
    asyncio.run(ingest_linear(parser.parse_args().days))
