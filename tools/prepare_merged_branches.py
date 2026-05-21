#!/usr/bin/env python3
"""
Collect PR-integrated branches from a target branch within a configurable UTC lookback window.

When GITHUB_OUTPUT and GITHUB_STEP_SUMMARY are present, this script writes the same
workflow outputs and summary content expected by dev_previous_day_commits.yml.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect merged branches from a git branch")
    parser.add_argument(
        "--branch", default="origin/dev", help="Git ref to inspect for merge commits"
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


def try_gh_head_ref(pr_number: str) -> str | None:
    if shutil.which("gh") is None:
        return None

    try:
        result = subprocess.check_output(
            ["gh", "pr", "view", pr_number, "--json", "headRefName", "--jq", ".headRefName"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return result or None


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


def try_github_api_head_ref(pr_number: str, repo: str | None) -> str | None:
    if not repo:
        return None

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "prepare_merged_branches.py",
        },
    )
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    head = payload.get("head")
    if isinstance(head, dict):
        ref = head.get("ref")
        if isinstance(ref, str) and ref.strip():
            return ref.strip()

    return None


def extract_subject_pr_number(subject: str) -> str | None:
    merge_subject_match = re.search(r"^Merge pull request #(\d+)\b", subject)
    if merge_subject_match:
        return merge_subject_match.group(1)

    squash_subject_match = re.search(r"\(#(\d+)\)$", subject.strip())
    if squash_subject_match:
        return squash_subject_match.group(1)

    return None


def extract_pr_number(subject: str, body: str) -> str | None:
    subject_pr_number = extract_subject_pr_number(subject)
    if subject_pr_number:
        return subject_pr_number

    for text in (subject, body):
        pr_match = re.search(r"#(\d+)", text)
        if pr_match:
            return pr_match.group(1)
    return None


def find_branch_ref(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        pr_match = re.search(
            r"^Merge pull request #\d+ from [^/\s]+/([A-Za-z0-9._/\-]+)$", stripped
        )
        if pr_match:
            return pr_match.group(1)

        from_match = re.search(r"^from [^/\s]+/([A-Za-z0-9._/\-]+)$", stripped)
        if from_match:
            return from_match.group(1)

    return None


def get_branch_name(subject: str, body: str, merge_sha: str, repo: str | None) -> str:
    pr_match = re.search(r"^Merge pull request #\d+ from [^/\s]+/([A-Za-z0-9._/\-]+)$", subject)
    if pr_match:
        return pr_match.group(1).strip()

    branch_match = re.search(r"Merge branch '([^']+)'", subject)
    if branch_match:
        return branch_match.group(1).strip()

    body_ref = find_branch_ref(body)
    if body_ref:
        return body_ref

    pr_number = extract_pr_number(subject, body)
    if pr_number:
        api_head_ref = try_github_api_head_ref(pr_number, repo)
        if api_head_ref:
            return api_head_ref

        head_ref = try_gh_head_ref(pr_number)
        if head_ref:
            return head_ref

    raise RuntimeError(
        f"Could not resolve branch name for commit {merge_sha} from subject/body metadata."
    )


def collect_merges(
    branch: str, lookback_days: int, repo: str | None, anchor_date: str | None = None
) -> dict[str, Any]:
    effective_lookback = max(1, lookback_days)
    start, end, window_label = resolve_time_window(effective_lookback, anchor_date)

    raw_log = git(
        "log",
        branch,
        "--first-parent",
        f"--since={start.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"--until={end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "--pretty=format:%H%x1f%P%x1f%s%x1f%b%x1e",
    )

    merges = []
    for record in raw_log.split("\x1e"):
        if not record.strip():
            continue
        merge_sha, parents, subject, body = record.strip().split("\x1f", 3)
        parent_parts = parents.split()
        if not parent_parts:
            continue

        pr_number = extract_subject_pr_number(subject)
        is_merge_commit = len(parent_parts) >= 2
        is_squash_merged_pr = len(parent_parts) == 1 and pr_number is not None
        if not is_merge_commit and not is_squash_merged_pr:
            continue

        branch_name = get_branch_name(subject, body, merge_sha, repo)
        safe_branch = re.sub(r"[^a-z0-9]+", "-", branch_name.lower()).strip("-")
        if not safe_branch:
            raise RuntimeError(
                f"Could not derive a safe branch slug from branch name {branch_name!r} for merge {merge_sha}."
            )

        first_parent = parent_parts[0]
        second_parent = parent_parts[1] if is_merge_commit else merge_sha
        merges.append(
            {
                "merge_sha": merge_sha,
                "short_sha": merge_sha[:7],
                "first_parent": first_parent,
                "second_parent": second_parent,
                "branch_name": branch_name,
                "safe_branch": safe_branch,
                "subject": subject,
            }
        )

    merge_lines = [
        f"- {item['branch_name']} ({item['short_sha']}): {item['subject']}" for item in merges
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
    return f"No branches were merged into {branch} during the {payload['window_label']}."


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
        fh.write("## Merged branches on dev\n\n")
        fh.write(f"- Source branch: `{payload['branch']}`\n")
        fh.write(f"- Lookback days: `{payload['lookback_days']}`\n")
        fh.write(f"- Time window (UTC): `{format_time_window(payload)}`\n\n")
        if payload["merge_lines"]:
            fh.write("### Merged branches\n\n")
        fh.write("\n".join(get_merge_summary_lines(payload, markdown_branch=True)))
        fh.write("\n")


def print_console_summary(payload: dict[str, Any]) -> None:
    print(f"Merged branches on {payload['branch']}")
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
