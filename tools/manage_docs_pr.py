#!/usr/bin/env python3
"""Create or update an automated draft PR in the docs repository."""

from __future__ import annotations

import argparse
import json
import os
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

    existing_prs = github_request(
        token,
        args.target_repo,
        "GET",
        "/pulls",
        query={
            "head": f"{owner}:{args.head_branch}",
            "base": args.base_branch,
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
        pr = github_request(
            token,
            args.target_repo,
            "POST",
            "/pulls",
            {
                "base": args.base_branch,
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

    write_pr_output(pr)


if __name__ == "__main__":
    main()
