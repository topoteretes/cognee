"""Remember the Linear issues changed in the last 30 days (node set `linear`).

Needs LINEAR_API_KEY, a personal API key (Linear: Settings → Security & access), in .env at
the repo root. With --sample, it reads the issues setup.py wrote instead.

Run alone: uv run python examples/cookbooks/company_brain/follow_up_agent/scripts/ingest_linear.py [--days N] [--sample]
"""

import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import httpx

import cognee  # also loads .env, so a LINEAR_API_KEY set there is seen

DATASET = "company_brain"  # the same in every script
SAMPLE = Path(__file__).parent.parent / "sample"

QUERY = """query($since: DateTimeOrDuration!, $after: String) {
  issues(filter: {updatedAt: {gt: $since}}, first: 250, after: $after) {
    nodes {
      identifier title description dueDate url
      state { name } assignee { name } team { name } project { name }
    }
    pageInfo { hasNextPage endCursor }
  }
}"""


def linear_issues(days: int, sample: bool = False) -> list[str]:
    """Linear issues changed in the last ``days`` days, each as text."""
    if sample:
        return [path.read_text() for path in sorted((SAMPLE / "linear").glob("*.txt"))]
    if not os.environ.get("LINEAR_API_KEY"):
        raise SystemExit("[ingest_linear] MISSING: LINEAR_API_KEY is not set (put it in .env).")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    issues, after = [], None
    while True:  # 250 issues a page, the most Linear returns
        response = httpx.post(
            "https://api.linear.app/graphql",
            json={"query": QUERY, "variables": {"since": since, "after": after}},
            headers={"Authorization": os.environ["LINEAR_API_KEY"]},  # a personal API key
            timeout=60,
        )
        page = response.raise_for_status().json()["data"]["issues"]
        issues.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    name = lambda field: (field or {}).get("name") or "none"
    return [
        f"Linear issue {i['identifier']}: {i['title']}\nStatus: {name(i['state'])}\n"
        f"Team: {name(i['team'])}\nProject: {name(i['project'])}\n"
        f"Assignee: {name(i['assignee'])}\nDue: {i['dueDate'] or 'none'}\n\n{i['description'] or ''}"
        for i in issues
    ]


async def ingest_linear(days: int = 30, sample: bool = False) -> None:
    issues = linear_issues(days, sample)
    if not issues:
        print(f"[ingest_linear] No Linear issues changed in the last {days} days.")
        return
    await cognee.remember(issues, dataset_name=DATASET, node_set=["linear"], self_improvement=False)
    source = "sample Linear issues" if sample else f"Linear issues from the last {days} days"
    print(f"[ingest_linear] Remembered {len(issues)} {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30, help="how far back to read issues")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    args = parser.parse_args()
    asyncio.run(ingest_linear(args.days, args.sample))
