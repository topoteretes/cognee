"""Ingest GitHub issues (title + body + comments) from a public repo into org dataset.

Pulls open + closed issues, skips PRs. For closed issues we also fetch the
comments thread (that's where the actual fix usually lives).

Usage:
    python -m community_bot.ingest.github_issues
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cognee  # noqa: E402

from config import (  # noqa: E402
    CODE_REPO_NAME,
    CODE_REPO_OWNER,
    GITHUB_TOKEN,
    MAX_ISSUES,
    ORG_DATASET,
)

GITHUB_API = "https://api.github.com"
PER_PAGE = 100


def _gh_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


async def _paginate_issues(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"{GITHUB_API}/repos/{CODE_REPO_OWNER}/{CODE_REPO_NAME}/issues"
            f"?state=all&per_page={PER_PAGE}&page={page}"
        )
        resp = await client.get(url, headers=_gh_headers())
        resp.raise_for_status()
        raw_batch = resp.json()
        # Stop only when the RAW response is short — GitHub returns PRs and
        # issues mixed together on this endpoint, so a PR-heavy page can
        # produce a small filtered batch while further pages still have real
        # issues on them.
        if not raw_batch:
            break
        issues.extend(i for i in raw_batch if "pull_request" not in i)
        if len(raw_batch) < PER_PAGE:
            break
        if 0 < MAX_ISSUES <= len(issues):
            break
        page += 1
    if 0 < MAX_ISSUES < len(issues):
        issues = issues[:MAX_ISSUES]
    return issues


async def _fetch_comments(client: httpx.AsyncClient, issue_number: int) -> list[dict[str, Any]]:
    url = (
        f"{GITHUB_API}/repos/{CODE_REPO_OWNER}/{CODE_REPO_NAME}"
        f"/issues/{issue_number}/comments?per_page=100"
    )
    resp = await client.get(url, headers=_gh_headers())
    if resp.status_code != 200:
        return []
    return resp.json()


def _format_issue(issue: dict[str, Any], comments: list[dict[str, Any]]) -> str:
    labels = ", ".join(lbl.get("name", "") for lbl in issue.get("labels", []))
    lines = [
        f"# GitHub Issue #{issue['number']}: {issue['title']}",
        f"State: {issue.get('state', 'unknown')}",
        f"Repo: {CODE_REPO_OWNER}/{CODE_REPO_NAME}",
        f"URL: {issue.get('html_url', '')}",
    ]
    if labels:
        lines.append(f"Labels: {labels}")
    if issue.get("user", {}).get("login"):
        lines.append(f"Reporter: {issue['user']['login']}")
    lines.append("")
    lines.append("## Body")
    lines.append((issue.get("body") or "(no body)").strip())

    if comments:
        lines.append("")
        lines.append("## Comments")
        for c in comments:
            author = c.get("user", {}).get("login", "unknown")
            body = (c.get("body") or "").strip()
            lines.append(f"- {author}: {body}")

    return "\n".join(lines)


async def ingest_issues() -> int:
    async with httpx.AsyncClient(timeout=60.0) as client:
        issues = await _paginate_issues(client)
        print(
            f"[issues] Pulled {len(issues)} issues from "
            f"{CODE_REPO_OWNER}/{CODE_REPO_NAME} (MAX_ISSUES={MAX_ISSUES})"
        )

        ingested = 0
        for issue in issues:
            # Only fetch comments for closed issues — that's where fixes live
            comments: list[dict[str, Any]] = []
            if issue.get("state") == "closed" and issue.get("comments", 0) > 0:
                try:
                    comments = await _fetch_comments(client, issue["number"])
                except httpx.HTTPError as exc:
                    print(f"[issues] ! failed to fetch comments for #{issue['number']}: {exc}")

            text = _format_issue(issue, comments)
            await cognee.add(text, dataset_name=ORG_DATASET)
            ingested += 1
            if ingested % 10 == 0 or ingested == len(issues):
                print(f"[issues] added {ingested}/{len(issues)}")

    print(f"[issues] Running cognify on '{ORG_DATASET}' ...")
    await cognee.cognify(datasets=[ORG_DATASET])
    print(f"[issues] Done. Ingested {ingested} issues.")
    return ingested


if __name__ == "__main__":
    asyncio.run(ingest_issues())
