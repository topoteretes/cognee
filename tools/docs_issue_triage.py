#!/usr/bin/env python3
"""Triage GitHub issues on topoteretes/cognee for documentation problems.

Phase 1 (this version) is read-only against GitHub: it selects open issues by
number or by created-at date range, runs the cheap docs filter from
tools/docs_issue_filter.py, and writes a run summary. It never comments on,
labels, or closes an issue. Later phases add the public-docs check, the source
check, the human-review email, and the cognee-docs draft PR.

Exit codes: 2 bad arguments, 1 GitHub/HTTP failure, 0 success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docs_issue_filter import docs_match_reason, looks_like_docs_issue

DEFAULT_REPO = "topoteretes/cognee"
GITHUB_API = "https://api.github.com"

# Full verdict vocabulary. Phase 1 only emits the first four; the rest are reserved
# so later phases extend the set instead of renaming it.
VERDICTS = (
    "skipped_pr",
    "skipped_closed",
    "skipped_filter",
    "pending_docs_check",
    "not_docs",
    "too_vague",
    "already_answered",
    "needs_source",
    "not_in_source",
    "too_big",
    "uncertain",
    "small_gap",
)

# Verdicts that feed the draft-docs matrix job (phase 3). Empty until then.
GAP_VERDICTS: tuple[str, ...] = ()


def github_api_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "docs_issue_triage.py",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _clean(value: str | None) -> str:
    return (value or "").strip()


def parse_utc_date(value: str, flag: str, parser: argparse.ArgumentParser) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        parser.error(f"{flag} must be a UTC date in YYYY-MM-DD form, got {value!r}")
    raise AssertionError("unreachable")  # pragma: no cover


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triage GitHub issues for documentation problems (read-only in phase 1)"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO,
        help="owner/name of the repository whose issues are triaged",
    )
    parser.add_argument("--issue-number", default="", help="One open issue number")
    parser.add_argument("--since", default="", help="UTC start date YYYY-MM-DD (created_at)")
    parser.add_argument(
        "--until", default="", help="UTC end date YYYY-MM-DD (created_at, inclusive)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Never comment or open docs PRs. Phase 1 performs no writes either way.",
    )
    parser.add_argument("--results-json", type=Path, default=None)
    args = parser.parse_args(argv)

    issue_number = _clean(args.issue_number)
    since = _clean(args.since)
    until = _clean(args.until)

    if issue_number:
        if not issue_number.isdigit():
            parser.error(f"--issue-number must be digits only, got {issue_number!r}")
        args.issue_number = int(issue_number)
        args.since = None
        args.until = None
        return args

    args.issue_number = None
    if not since or not until:
        parser.error("provide --issue-number, or both --since and --until")
    args.since = parse_utc_date(since, "--since", parser)
    args.until = parse_utc_date(until, "--until", parser)
    if args.since > args.until:
        parser.error("--since must not be later than --until")
    return args


def parse_github_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def make_result(issue: dict[str, Any], verdict: str, reason: str) -> dict[str, Any]:
    assert verdict in VERDICTS, verdict
    return {
        "number": issue.get("number"),
        "title": issue.get("title") or "",
        "html_url": issue.get("html_url") or "",
        "verdict": verdict,
        "reason": reason,
        "doc_urls": [],
        "source_files": [],
        "docs_files": [],
        "commented": False,
    }


def classify_open_issue(issue: dict[str, Any]) -> dict[str, Any]:
    labels = issue.get("labels") or []
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    reason = docs_match_reason(labels, title, body)
    if looks_like_docs_issue(labels, title, body):
        return make_result(issue, "pending_docs_check", reason)
    return make_result(issue, "skipped_filter", reason)


def select_single_issue(repo: str, number: int) -> list[dict[str, Any]]:
    issue = github_api_json(f"{GITHUB_API}/repos/{repo}/issues/{number}")
    if "pull_request" in issue:
        return [make_result(issue, "skipped_pr", "payload is a pull request")]
    if issue.get("state") != "open":
        return [make_result(issue, "skipped_closed", f"issue state is {issue.get('state')!r}")]
    return [classify_open_issue(issue)]


def select_issues_created_between(repo: str, since: date, until: date) -> list[dict[str, Any]]:
    window_start = datetime(since.year, since.month, since.day, 0, 0, 0, tzinfo=timezone.utc)
    window_end = datetime(until.year, until.month, until.day, 23, 59, 59, tzinfo=timezone.utc)

    results: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {"state": "open", "per_page": "100", "page": str(page), "sort": "created"}
        )
        batch = github_api_json(f"{GITHUB_API}/repos/{repo}/issues?{query}")
        if not batch:
            break
        for issue in batch:
            if "pull_request" in issue:
                continue
            created_at = parse_github_timestamp(issue["created_at"])
            if not (window_start <= created_at <= window_end):
                continue
            results.append(classify_open_issue(issue))
        page += 1

    results.sort(key=lambda row: row["number"] or 0)
    return results


def summary_table_lines(results: list[dict[str, Any]]) -> list[str]:
    lines = ["| Issue | Verdict | Reason | Commented |", "|---|---|---|---|"]
    for row in results:
        number = row["number"]
        link = f"[#{number}]({row['html_url']})" if row["html_url"] else f"#{number}"
        title = row["title"].replace("|", "\\|")
        reason = row["reason"].replace("|", "\\|")
        commented = "yes" if row["commented"] else "no"
        lines.append(f"| {link} {title} | `{row['verdict']}` | {reason} | {commented} |")
    if len(results) == 0:
        lines.append("| _no open issues selected_ | | | |")
    return lines


def describe_selector(args: argparse.Namespace) -> str:
    if args.issue_number is not None:
        return f"issue #{args.issue_number}"
    return f"open issues created {args.since.isoformat()} .. {args.until.isoformat()} (UTC)"


def write_step_summary(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "## Docs issue triage",
        "",
        f"- Repository: `{args.repo}`",
        f"- Selector: {describe_selector(args)}",
        f"- Dry run: `{'true' if args.dry_run else 'false'}`",
        f"- Issues selected: {len(results)}",
        "",
        "Phase 1 is read-only: no comments, no docs PRs, no email.",
        "",
        *summary_table_lines(results),
        "",
    ]
    with Path(summary_path).open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_github_output(results: list[dict[str, Any]], dry_run: bool) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    gap_rows = [row for row in results if row["verdict"] in GAP_VERDICTS] if not dry_run else []
    with Path(github_output).open("a", encoding="utf-8") as fh:
        fh.write(f"has_gaps={'true' if gap_rows else 'false'}\n")
        fh.write(f"matrix={json.dumps(gap_rows)}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.issue_number is not None:
            results = select_single_issue(args.repo, args.issue_number)
        else:
            results = select_issues_created_between(args.repo, args.since, args.until)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(
            f"GitHub API request failed: {exc.code} {exc.reason} {exc.url}\n{detail}",
            file=sys.stderr,
        )
        return 1
    except urllib.error.URLError as exc:
        print(f"GitHub API request failed: {exc.reason}", file=sys.stderr)
        return 1

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Selector: {describe_selector(args)}")
    print("\n".join(summary_table_lines(results)))
    write_step_summary(args, results)
    write_github_output(results, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
