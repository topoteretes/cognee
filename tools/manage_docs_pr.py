#!/usr/bin/env python3
"""Create or update an automated draft PR in the docs repository.

With --daily-date, a new PR targets that day's daily automation branch in the
docs repo (automation/daily-YYYY-MM-DD) instead of --base-branch, creating the
branch and its PR into main first if no automation has yet (RES-91). This is a
copy of `ensure` and `refresh` in cognee-docs' .github/actions/daily-pr/daily_pr.py,
which this workflow cannot use because cognee-docs is private: keep the branch
name, title, body and empty first commit in step with it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a docs draft PR")
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--head-branch", required=True)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--pr-title", required=True)
    parser.add_argument("--pr-body", required=True)
    parser.add_argument("--changes-made", required=True, choices=["true", "false"])
    parser.add_argument(
        "--daily-date",
        default="",
        help="YYYY-MM-DD: open a new PR against that day's daily automation branch",
    )
    return parser.parse_args()


def github_request(
    token: str,
    target_repo: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
) -> Any:
    url = f"https://api.github.com/repos/{target_repo}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"), flush=True)
        raise
    return json.loads(content) if content else None


DAILY_BRANCH_PREFIX = "automation/daily-"
DAILY_BASE = "main"
DAILY_BODY_MARKER = (
    "<!-- daily-pr: maintained by .github/actions/daily-pr; manual edits are overwritten -->"
)


def get_branch_sha(token: str, repo: str, branch: str) -> str | None:
    try:
        ref = github_request(token, repo, "GET", f"/git/ref/heads/{urllib.parse.quote(branch)}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return ref["object"]["sha"]


def find_daily_pr(token: str, repo: str, branch: str) -> dict[str, Any] | None:
    owner = repo.split("/", 1)[0]
    prs = github_request(
        token,
        repo,
        "GET",
        "/pulls",
        query={"head": f"{owner}:{branch}", "base": DAILY_BASE, "state": "open"},
    )
    return min(prs, key=lambda pr: pr["number"]) if prs else None


def close_duplicate_daily_prs(token: str, repo: str, branch: str, created: dict[str, Any]) -> None:
    """Parallel jobs can each open a daily PR: GitHub does not always reject a
    second PR with the same head and base when both arrive at once. Every job
    keeps the oldest and closes the one it opened if that lost."""
    time.sleep(3)
    keep = find_daily_pr(token, repo, branch)
    if keep is not None and keep["number"] != created["number"]:
        github_request(
            token,
            repo,
            "POST",
            f"/issues/{created['number']}/comments",
            {"body": f"Duplicate of #{keep['number']}, opened by a parallel automation run."},
        )
        github_request(token, repo, "PATCH", f"/pulls/{created['number']}", {"state": "closed"})


def empty_commit_on_main(token: str, repo: str, date: str) -> str:
    """GitHub refuses a PR with no commits, so the daily branch starts with an
    empty one: same tree as main, main as its parent."""
    main_sha = get_branch_sha(token, repo, DAILY_BASE)
    tree = github_request(token, repo, "GET", f"/git/commits/{main_sha}")["tree"]["sha"]
    commit = github_request(
        token,
        repo,
        "POST",
        "/git/commits",
        {
            "message": f"chore: start daily automation branch {date}",
            "tree": tree,
            "parents": [main_sha],
        },
    )
    return commit["sha"]


def prepare_daily_branch(token: str, repo: str, branch: str, date: str) -> None:
    if get_branch_sha(token, repo, branch) is None:
        try:
            github_request(
                token,
                repo,
                "POST",
                "/git/refs",
                {"ref": f"refs/heads/{branch}", "sha": empty_commit_on_main(token, repo, date)},
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 422:  # 422: a parallel job created it first
                raise
        return
    # The branch exists but has no open PR: its PR was merged without the branch
    # being deleted. Nothing on it is missing from main, so restart it there.
    compare = github_request(
        token, repo, "GET", f"/compare/{DAILY_BASE}...{urllib.parse.quote(branch)}"
    )
    if compare["ahead_by"] == 0:
        github_request(
            token,
            repo,
            "PATCH",
            f"/git/refs/heads/{urllib.parse.quote(branch)}",
            {"sha": empty_commit_on_main(token, repo, date), "force": True},
        )


def render_daily_body(date: str, children: list[int]) -> str:
    branch = f"{DAILY_BRANCH_PREFIX}{date}"
    lines = [
        f"## Daily docs automation: {date} (UTC)",
        "",
        f"The docs automations open their PRs against `{branch}` instead of `main`.",
        "",
        "1. Review each child PR below and merge the ones you want into this branch.",
        (
            "2. Merge this PR into `main` with a **merge commit**, not squash or rebase. "
            "Squashing would make the children that are still open show this PR's changes again once they move to `main`."
        ),
        "3. Child PRs still open when this PR merges are retargeted to `main`.",
        "",
        "The first commit on this branch is empty: GitHub cannot open a PR without one.",
        "",
        "### Child PRs",
        "",
    ]
    lines += [f"- #{number}" for number in children] or ["_None yet._"]
    lines += ["", DAILY_BODY_MARKER]
    return "\n".join(lines)


def ensure_daily_pr(token: str, repo: str, date: str) -> str:
    """Return the daily branch for `date`, opening it and its PR if missing."""
    datetime.date.fromisoformat(date)  # reject anything else before it reaches a ref name
    branch = f"{DAILY_BRANCH_PREFIX}{date}"
    if find_daily_pr(token, repo, branch) is None:
        prepare_daily_branch(token, repo, branch, date)
        try:
            pr = github_request(
                token,
                repo,
                "POST",
                "/pulls",
                {
                    "base": DAILY_BASE,
                    "head": branch,
                    "title": f"docs: daily automation {date}",
                    "body": render_daily_body(date, []),
                },
            )
            github_request(
                token,
                repo,
                "POST",
                f"/issues/{pr['number']}/labels",
                {"labels": ["documentation", "automated"]},
            )
            close_duplicate_daily_prs(token, repo, branch, pr)
        except urllib.error.HTTPError as exc:
            if exc.code != 422 or find_daily_pr(token, repo, branch) is None:
                raise  # 422 with a PR found: a parallel job opened it first
    return branch


def refresh_daily_pr(token: str, repo: str, date: str) -> None:
    """Rewrite the daily PR body with the current list of child PRs."""
    branch = f"{DAILY_BRANCH_PREFIX}{date}"
    pr = find_daily_pr(token, repo, branch)
    if pr is None:
        return
    body = pr["body"] or ""
    # Jobs refresh in parallel. Re-list after writing, so a child opened while
    # this job was writing is not left out by a body computed before it existed.
    for _ in range(3):
        children = [
            child["number"]
            for child in github_request(
                token,
                repo,
                "GET",
                "/pulls",
                query={
                    "base": branch,
                    "state": "all",
                    "per_page": "100",
                    "sort": "created",
                    "direction": "asc",
                },
            )
        ]
        wanted = render_daily_body(date, children)
        if wanted == body:
            return
        github_request(token, repo, "PATCH", f"/pulls/{pr['number']}", {"body": wanted})
        body = wanted
        time.sleep(2)


def write_pr_output(pr: dict[str, Any] | None) -> None:
    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as fh:
        if pr is None:
            fh.write("pr_number=\n")
            fh.write("pr_url=\n")
        else:
            fh.write(f"pr_number={pr['number']}\n")
            fh.write(f"pr_url={pr['html_url']}\n")


def main() -> None:
    args = parse_args()
    token = os.environ["GH_TOKEN"]
    owner = args.target_repo.split("/", 1)[0]
    changes_made = args.changes_made == "true"

    # Any base: a PR opened before the daily branch existed keeps its base.
    existing_prs = github_request(
        token,
        args.target_repo,
        "GET",
        "/pulls",
        query={
            "head": f"{owner}:{args.head_branch}",
            "state": "open",
            "per_page": "1",
        },
    )

    if existing_prs:
        pr = github_request(
            token,
            args.target_repo,
            "PATCH",
            f"/pulls/{existing_prs[0]['number']}",
            {"title": args.pr_title, "body": args.pr_body},
        )
    elif changes_made:
        base_branch = args.base_branch
        if args.daily_date:
            base_branch = ensure_daily_pr(token, args.target_repo, args.daily_date)
        pr = github_request(
            token,
            args.target_repo,
            "POST",
            "/pulls",
            {
                "base": base_branch,
                "head": args.head_branch,
                "title": args.pr_title,
                "body": args.pr_body,
                "draft": True,
            },
        )
    else:
        print(
            f"Assessment expected documentation changes for {args.head_branch}, "
            "but Claude made no docs edits and no open PR exists. Skipping PR creation."
        )
        pr = None

    if pr is not None and pr["base"]["ref"].startswith(DAILY_BRANCH_PREFIX):
        refresh_daily_pr(token, args.target_repo, pr["base"]["ref"][len(DAILY_BRANCH_PREFIX) :])

    write_pr_output(pr)


if __name__ == "__main__":
    main()
