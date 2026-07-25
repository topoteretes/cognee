#!/usr/bin/env python3
"""
Collect PRs merged into a target branch within a configurable UTC lookback window.

When GITHUB_OUTPUT and GITHUB_STEP_SUMMARY are present, this script writes the same
workflow outputs and summary content expected by dev_previous_day_commits.yml.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PRs merged into a git branch")
    parser.add_argument(
        "--branch", default="origin/dev", help="Target branch to inspect for merged PRs"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1,
        help="Number of UTC calendar days to look back, including today",
    )
    parser.add_argument(
        "--anchor-date",
        default=None,
        help="UTC date to anchor the scan window, in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repository in owner/repo form. Defaults to parsing origin remote.",
    )
    return parser.parse_args()


def parse_anchor_date(anchor_date: str | None) -> date | None:
    if anchor_date is None:
        return None

    try:
        return datetime.strptime(anchor_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid --anchor-date value {anchor_date!r}. Expected YYYY-MM-DD."
        ) from exc


def resolve_time_window(
    lookback_days: int, anchor_date: str | None
) -> tuple[datetime, datetime, str]:
    effective_lookback = max(1, lookback_days)
    anchor_day = parse_anchor_date(anchor_date)

    if anchor_day is not None:
        start_date = anchor_day - timedelta(days=effective_lookback - 1)
        start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
        end = datetime.combine(anchor_day, time(23, 59, 59), tzinfo=timezone.utc)
        window_label = (
            f"UTC day {anchor_day.isoformat()}"
            if effective_lookback == 1
            else f"last {effective_lookback} UTC days ending on {anchor_day.isoformat()}"
        )
        return start, end, window_label

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=effective_lookback - 1)
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = now
    window_label = (
        "current UTC day" if effective_lookback == 1 else f"last {effective_lookback} UTC days"
    )
    return start, end, window_label


def parse_github_repo(repo_url: str) -> str | None:
    ssh_match = re.match(r"git@github\.com:([^/]+/[^/]+?)(?:\.git)?$", repo_url)
    if ssh_match:
        return ssh_match.group(1)

    https_match = re.match(r"https://github\.com/([^/]+/[^/]+?)(?:\.git)?$", repo_url)
    if https_match:
        return https_match.group(1)

    return None


def get_github_repo(explicit_repo: str | None) -> str | None:
    if explicit_repo:
        return explicit_repo

    try:
        remote_url = git("remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        return None

    return parse_github_repo(remote_url)


def github_api_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "prepare_merged_branches.py",
        },
    )
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def search_merged_pr_numbers(
    repo: str, base_branch: str, start: datetime, end: datetime
) -> list[int]:
    from urllib.parse import quote

    query = " ".join(
        [
            f"repo:{repo}",
            "is:pr",
            "is:merged",
            f"base:{base_branch}",
            f"merged:{start.date().isoformat()}..{end.date().isoformat()}",
        ]
    )
    numbers: list[int] = []
    for page in range(1, 11):
        url = (
            "https://api.github.com/search/issues"
            f"?q={quote(query)}&sort=updated&order=desc&per_page=100&page={page}"
        )
        payload = github_api_json(url)
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if not items:
            break

        for item in items:
            if not isinstance(item, dict):
                continue
            number = item.get("number")
            if isinstance(number, int):
                numbers.append(number)

        if len(items) < 100:
            break

    return numbers


def get_pull_request(repo: str, pr_number: int) -> dict[str, Any]:
    payload = github_api_json(f"https://api.github.com/repos/{repo}/pulls/{pr_number}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected GitHub API response for PR #{pr_number}.")
    return payload


def parse_github_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def get_commit_parents(commit_sha: str) -> list[str]:
    line = git("rev-list", "--parents", "-n", "1", commit_sha)
    parts = line.split()
    return parts[1:]


def resolve_pr_diff_range(merge_sha: str) -> tuple[str, str]:
    parents = get_commit_parents(merge_sha)
    if len(parents) >= 2:
        return parents[0], parents[1]
    if len(parents) == 1:
        return parents[0], merge_sha
    raise RuntimeError(f"Could not resolve parents for merged PR commit {merge_sha}.")


def collect_merges(
    branch: str, lookback_days: int, repo: str | None, anchor_date: str | None = None
) -> dict[str, Any]:
    effective_lookback = max(1, lookback_days)
    start, end, window_label = resolve_time_window(effective_lookback, anchor_date)

    github_repo = get_github_repo(repo)
    if not github_repo:
        raise RuntimeError(
            "Could not determine GitHub repository. Pass --repo owner/repo or configure origin."
        )

    base_branch = branch.removeprefix("origin/")
    merges = []
    for pr_number in search_merged_pr_numbers(github_repo, base_branch, start, end):
        pr = get_pull_request(github_repo, pr_number)
        merged_at = parse_github_timestamp(pr.get("merged_at"))
        if merged_at is None or not (start <= merged_at <= end):
            continue

        merge_sha = pr.get("merge_commit_sha")
        if not isinstance(merge_sha, str) or not merge_sha:
            continue

        first_parent, second_parent = resolve_pr_diff_range(merge_sha)
        head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
        branch_name = str(head.get("ref") or f"pr-{pr_number}")
        safe_branch = re.sub(r"[^a-z0-9]+", "-", branch_name.lower()).strip("-")
        if not safe_branch:
            raise RuntimeError(
                f"Could not derive a safe branch slug from branch name {branch_name!r} for PR #{pr_number}."
            )

        title = str(pr.get("title") or f"PR #{pr_number}")
        body = str(pr.get("body") or "")
        body_b64 = base64.b64encode(body[:4000].encode("utf-8")).decode("ascii")
        merges.append(
            {
                "pr_number": pr_number,
                "pr_title": title,
                "pr_body_b64": body_b64,
                "pr_url": pr.get("html_url") or "",
                "merged_at": merged_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "merge_sha": merge_sha,
                "short_sha": merge_sha[:7],
                "first_parent": first_parent,
                "second_parent": second_parent,
                "branch_name": branch_name,
                "safe_branch": safe_branch,
                "subject": f"PR #{pr_number}: {title}",
            }
        )

    merges.sort(key=lambda item: item["merged_at"])

    merge_lines = [
        f"- #{item['pr_number']} {item['branch_name']} ({item['short_sha']}): {item['pr_title']}"
        for item in merges
    ]

    return {
        "branch": branch,
        "lookback_days": effective_lookback,
        "anchor_date": anchor_date,
        "start": start,
        "end": end,
        "window_label": window_label,
        "merges": merges,
        "merge_lines": merge_lines,
    }


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def format_time_window(payload: dict[str, Any]) -> str:
    return f"{format_timestamp(payload['start'])} to {format_timestamp(payload['end'])}"


def format_no_merges_message(payload: dict[str, Any], markdown_branch: bool = False) -> str:
    branch = f"`{payload['branch']}`" if markdown_branch else payload["branch"]
    return f"No PRs were merged into {branch} during the {payload['window_label']}."


def get_merge_summary_lines(payload: dict[str, Any], markdown_branch: bool = False) -> list[str]:
    if payload["merge_lines"]:
        return payload["merge_lines"]
    return [format_no_merges_message(payload, markdown_branch=markdown_branch)]


def write_github_output(payload: dict[str, Any]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return

    output_path = Path(github_output)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"start_date={format_timestamp(payload['start'])}\n")
        fh.write(f"end_date={format_timestamp(payload['end'])}\n")
        fh.write(f"has_merges={'true' if payload['merges'] else 'false'}\n")
        fh.write(f"matrix={json.dumps(payload['merges'])}\n")
        fh.write("merge_summary<<EOF\n")
        fh.write("\n".join(get_merge_summary_lines(payload)))
        fh.write("\n")
        fh.write("EOF\n")


def write_github_summary(payload: dict[str, Any]) -> None:
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not github_summary:
        return

    summary_path = Path(github_summary)
    with summary_path.open("a", encoding="utf-8") as fh:
        fh.write("## Merged PRs on dev\n\n")
        fh.write(f"- Source branch: `{payload['branch']}`\n")
        fh.write(f"- Lookback days: `{payload['lookback_days']}`\n")
        fh.write(f"- Time window (UTC): `{format_time_window(payload)}`\n\n")
        if payload["merge_lines"]:
            fh.write("### Merged PRs\n\n")
        fh.write("\n".join(get_merge_summary_lines(payload, markdown_branch=True)))
        fh.write("\n")


def print_console_summary(payload: dict[str, Any]) -> None:
    print(f"Merged PRs on {payload['branch']}")
    print(f"Lookback days: {payload['lookback_days']}")
    print(f"Time window (UTC): {format_time_window(payload)}")
    for line in get_merge_summary_lines(payload):
        print(line)


def main() -> None:
    args = parse_args()
    payload = collect_merges(
        args.branch,
        args.lookback_days,
        get_github_repo(args.repo),
        anchor_date=args.anchor_date,
    )
    print_console_summary(payload)
    write_github_output(payload)
    write_github_summary(payload)


if __name__ == "__main__":
    main()
