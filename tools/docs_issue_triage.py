#!/usr/bin/env python3
"""Triage GitHub issues on topoteretes/cognee for documentation problems.

Phase 1 selects open issues by number or by created-at date range and runs the
cheap docs filter from tools/docs_issue_filter.py. Phase 2 (this version) takes
every issue that passed the filter, ranks the public docs pages against it with a
lexical (BM25) index built once per run from the site's full-text export, hands
the top pages to one structured LLM call that decides whether the site already
answers the report, and on ``documentation_covered`` posts a single marked comment
linking those pages and asking the author to close. Issues filed through the bug form are ``not_docs`` without
an LLM call: they ask for a code change. Phase 3 (this version) takes every ``needs_source``
issue to the cognee source: a ``git grep`` for the identifiers the issue names and a
second structured LLM call decide whether the product really has the behaviour and the
docs miss a small fact. An issue a maintainer has already replied on is left to them before any of this
runs. Confirmed gaps become matrix rows for the workflow's draft-docs
job (a Claude edit on topoteretes/cognee-docs, then ``--post-gap-comment``); everything
uncertain, too big, absent from source, or already being fixed goes to a human-review
list that is emailed when the SMTP secrets exist. It never labels or closes an issue.

Exit codes: 2 bad arguments, 1 GitHub/HTTP/LLM failure, 0 success.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import re
import smtplib
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docs_issue_filter import docs_signals, docs_site_urls, label_names, relevant_body_text

DEFAULT_REPO = "topoteretes/cognee"
GITHUB_API = "https://api.github.com"

# Full verdict vocabulary. Phase 1 only emits the first four; the rest are reserved
# so later phases extend the set instead of renaming it.
VERDICTS = (
    "skipped_pr",
    "skipped_closed",
    "skipped_filter",
    "pending_docs_check",
    # A maintainer already replied on the issue; the bot has nothing to add and leaves the
    # thread to them. Decided before any LLM call. Not in the original plan.
    "maintainer_replied",
    "not_docs",
    "too_vague",
    "documentation_covered",
    "needs_source",
    "not_in_source",
    "too_big",
    "uncertain",
    "small_gap",
    # Not in the original plan: an open maintainer PR already fixes the issue, so a docs edit
    # describing today's behaviour would be undone. Routed to human review.
    "fix_in_progress",
)

# Verdicts that feed the draft-docs matrix job (phase 3).
GAP_VERDICTS: tuple[str, ...] = ("small_gap",)
# Verdicts a human has to look at; emailed when the SMTP secrets are configured.
# maintainer_replied is deliberately absent: a maintainer already has the thread.
HUMAN_REVIEW_VERDICTS: tuple[str, ...] = (
    "not_in_source",
    "too_big",
    "uncertain",
    "fix_in_progress",
)

DOCS_SITE = "https://docs.cognee.ai"
# The full-site export, fetched ONCE per run (a few MB) and used only to rank pages
# lexically. Page text goes to the LLM only for the handful of pages picked per issue.
LLMS_FULL_URL = f"{DOCS_SITE}/llms-full.txt"
# Every page in the export starts with "# Title" followed by "Source: <page url>".
PAGE_HEADER_RE = re.compile(r"^# ([^\n]+)\nSource: (https://docs\.cognee\.ai/\S+)\n", re.MULTILINE)
# Pages that mention everything and answer nothing (release notes) or document the
# Rust/TypeScript ports; the ports come back as candidates only when the issue names them.
EXCLUDED_PAGE_RE = re.compile(r"https://docs\.cognee\.ai/(?:changelog|llms)")
PORT_PAGE_PREFIXES = {"rust": f"{DOCS_SITE}/rust/", "typescript": f"{DOCS_SITE}/typescript/"}

TOKEN_RE = re.compile(r"[a-z0-9]{4,}")
# Tokens that appear across the whole site and carry no topical signal.
TOKEN_STOPWORDS = frozenset(
    {
        "cognee",
        "with",
        "from",
        "this",
        "that",
        "your",
        "when",
        "have",
        "does",
        "into",
        "using",
        "overview",
        "guide",
        "docs",
        "documentation",
        "issue",
        "bugs",
        "feature",
        "should",
        "would",
        "could",
        "about",
        "which",
        "there",
        "their",
        "these",
        "those",
        "after",
        "before",
        "because",
        "every",
        "other",
        "some",
        "than",
        "then",
        "them",
        "will",
        "what",
        "where",
        "also",
        "only",
        "same",
        "each",
        "here",
        "just",
        "like",
        "reference",
        "introduction",
        "getting",
        "started",
        "https",
        "http",
        "theme",
        "null",
    }
)
BM25_K1 = 1.5
BM25_B = 0.75

MAX_PAGES = 4
# Candidate pages listed for the LLM (titles + URLs only), so it can say which one it used.
MAX_LISTED_CANDIDATES = 10
# Pages scoring below this fraction of the best page are noise, not alternatives.
MIN_RELATIVE_SCORE = 0.3
PAGE_MAX_CHARS = 20_000
ISSUE_BODY_MAX_CHARS = 8_000
SITE_CHECK_PROMPT = "docs_issue_site_check.txt"
SITE_CHECK_VERDICTS = ("not_docs", "too_vague", "documentation_covered", "needs_source")
# Issues filed through the bug form ask for a code change by definition: no docs page can
# settle them and no docs edit should describe the broken behaviour. They are not_docs
# before any LLM call.
BUG_TITLE_RE = re.compile(r"^\s*\[\s*bug\s*\]", re.IGNORECASE)
BUG_LABEL = "bug"

COMMENT_MARKER = "<!-- cognee-docs-issue-triage -->"

# --- phase 3 ---
SOURCE_CHECK_PROMPT = "docs_issue_source_check.txt"
SOURCE_CHECK_VERDICTS = ("not_in_source", "too_big", "uncertain", "small_gap")
REPO_ROOT = Path(__file__).resolve().parent.parent
GREP_PATHS = ("cognee", ".env.template")
GREP_EXCLUDES = (":(exclude)cognee/tests",)
GREP_MAX_CHARS = 80_000
GREP_MAX_TOKENS = 24
GREP_MIN_TOKEN_LEN = 3
IDENTIFIER_RES = (
    re.compile(r"`([^`\n]{3,80})`"),  # inline code
    re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b"),  # CONSTANTS / ENV_VARS
    re.compile(r"\bSearchType\.\w+"),
)
MAX_DOCS_FILES = 3
# Matrix rows carry the issue text base64-encoded, truncated like pr_body_b64 in
# tools/prepare_merged_branches.py; the draft-docs job decodes it to issue_excerpt.md.
MATRIX_EXCERPT_MAX_BYTES = 4000
# Comments whose author GitHub marks as an org MEMBER/OWNER, or whose login is in
# .github/core-team.txt, come from maintainers. A maintainer reply hands the thread to
# them (maintainer_replied); anyone else's comment changes nothing.
MAINTAINER_ASSOCIATIONS = frozenset({"MEMBER", "OWNER"})
CORE_TEAM_FILE = Path(".github") / "core-team.txt"
GAP_COMMENT = (
    f"{COMMENT_MARKER}\n"
    "This comment is auto-generated.\n\n"
    "Thanks for the report. It may be taken into account, and a documentation change may be "
    "prepared. A docs PR can still be rejected, so please do not assume the site has changed yet.\n"
)


def github_api_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "docs_issue_triage.py",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if data is not None:
        request.add_header("Content-Type", "application/json")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=20) as response:
        content = response.read().decode("utf-8")
    return json.loads(content) if content else None


def fetch_text(url: str, max_chars: int | None = None) -> str:
    """GET a public docs page as text. Raises urllib errors on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": "docs_issue_triage.py"})
    with urllib.request.urlopen(request, timeout=20) as response:
        text = response.read().decode("utf-8", errors="replace")
    return text[:max_chars] if max_chars else text


def _clean(value: str | None) -> str:
    return (value or "").strip()


def parse_utc_date(value: str, flag: str, parser: argparse.ArgumentParser) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        parser.error(f"{flag} must be a UTC date in YYYY-MM-DD form, got {value!r}")
    raise AssertionError("unreachable")  # pragma: no cover


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage GitHub issues for documentation problems")
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
        help="Never comment or open docs PRs; still fetch docs, call the LLM and write the summary.",
    )
    parser.add_argument("--results-json", type=Path, default=None)
    parser.add_argument(
        "--post-gap-comment",
        type=int,
        default=None,
        metavar="ISSUE",
        help=(
            "Phase 3 only: post the defensive 'a docs change may be prepared' comment on this "
            "issue and exit. Ignores the selectors; respects --dry-run and the marker. No LLM."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="cognee checkout to git-grep for the source check (default: this script's repo)",
    )
    args = parser.parse_args(argv)

    if args.post_gap_comment is not None:
        args.issue_number = None
        args.since = None
        args.until = None
        return args

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


def make_result(
    issue: dict[str, Any], verdict: str, reason: str, signals: list[str] | None = None
) -> dict[str, Any]:
    assert verdict in VERDICTS, verdict
    return {
        "number": issue.get("number"),
        "title": issue.get("title") or "",
        "html_url": issue.get("html_url") or "",
        "verdict": verdict,
        "reason": reason,
        "signals": list(signals or []),
        "pages_shown": [],
        # Pages that cover the report: {"title", "url", "note"}; doc_urls mirrors their URLs.
        "doc_pages": [],
        "doc_urls": [],
        "source_files": [],
        "docs_files": [],
        "commented": False,
        # The comment text: posted when ``commented`` is true, otherwise the comment a live
        # run would have posted (dry run, or suppressed by the marker). Empty when no verdict
        # calls for a comment. Lets a dry run be reviewed for exactly what it would say.
        "comment": "",
        # Not part of the published row; carried so phase 2 can read them without refetching.
        "_body": issue.get("body") or "",
        "_labels": label_names(issue.get("labels") or []),
    }


def classify_open_issue(issue: dict[str, Any]) -> dict[str, Any]:
    labels = issue.get("labels") or []
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    signals = docs_signals(labels, title, body)
    if signals:
        return make_result(issue, "pending_docs_check", "; ".join(signals), signals)
    return make_result(issue, "skipped_filter", "no documentation signal")


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
    """Markdown table of every row except the ones the cheap filter skipped.

    Those are the bulk of any backlog pass and carry no decision worth reading; the
    caller reports their count in one line instead.
    """
    listed = [row for row in results if row["verdict"] != "skipped_filter"]
    lines = [
        "| Issue | Signals | Verdict | Reason | Pages shown to the LLM | Comment |",
        "|---|---|---|---|---|---|",
    ]
    for row in listed:
        number = row["number"]
        link = f"[#{number}]({row['html_url']})" if row["html_url"] else f"#{number}"
        title = _cell(row["title"])
        signals = _cell("; ".join(row["signals"])) if row["signals"] else ""
        reason = _cell(row["reason"])
        cited = set(row["doc_urls"])
        pages = "<br>".join(
            f"{'**' if url in cited else ''}{url.removeprefix(DOCS_SITE + '/')}"
            f"{' (cited)**' if url in cited else ''}"
            for url in row["pages_shown"]
        )
        comment = _comment_cell(row)
        lines.append(
            f"| {link} {title} | {signals} | `{row['verdict']}` | {reason} | {pages} | {comment} |"
        )
    if not listed:
        lines.append("| _no issue passed the cheap filter_ | | | | | |")
    return lines


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _comment_cell(row: dict[str, Any]) -> str:
    """The comment as posted, or as a live run would have posted it, minus the marker line."""
    if row["verdict"] in GAP_VERDICTS:
        files = ", ".join(row["docs_files"]) or "?"
        return f"**draft-docs job**: edit {files}; gap comment only after the docs PR exists"
    if not row["comment"]:
        return ""
    body = "<br>".join(
        line.replace("|", "\\|")
        for line in row["comment"].splitlines()
        if line.strip() and line.strip() != COMMENT_MARKER
    )
    label = "**posted**" if row["commented"] else "**suggested, not posted**"
    return f"{label}<br>{body}"


def summary_count_lines(results: list[dict[str, Any]]) -> list[str]:
    skipped = sum(1 for row in results if row["verdict"] == "skipped_filter")
    listed = len(results) - skipped
    return [
        f"- Issues selected: {len(results)}",
        f"- Passed the cheap filter and listed below: {listed}",
        f"- No documentation signal, not listed: {skipped}",
    ]


def describe_selector(args: argparse.Namespace) -> str:
    if args.issue_number is not None:
        return f"issue #{args.issue_number}"
    return f"open issues created {args.since.isoformat()} .. {args.until.isoformat()} (UTC)"


# --- phase 2: public docs check ---------------------------------------------------------------


def stem(token: str) -> str:
    """Plural-stripping only: types/type, providers/provider, embeddings/embedding."""
    if len(token) >= 5 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # split CamelCase
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if stem(t) not in TOKEN_STOPWORDS]


def normalize_docs_url(url: str) -> str:
    """Comparable form of a docs.cognee.ai URL: no fragment, no query, no .md, no trailing /."""
    url = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    return url.removesuffix(".md").lower()


class DocsIndex:
    """BM25 over the pages of the full-site export."""

    def __init__(self, export_text: str) -> None:
        self.pages: dict[str, tuple[str, str]] = {}  # url -> (title, text)
        self._tf: dict[str, dict[str, int]] = {}
        self._length: dict[str, int] = {}
        headers = list(PAGE_HEADER_RE.finditer(export_text))
        for index, header in enumerate(headers):
            title, url = header.group(1).strip(), header.group(2).strip()
            end = headers[index + 1].start() if index + 1 < len(headers) else len(export_text)
            text = export_text[header.end() : end].strip()
            if url in self.pages:
                continue
            path = url.removeprefix(DOCS_SITE).replace("-", " ").replace("/", " ")
            tokens = tokenize(f"{title} {path}") + tokenize(text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.pages[url] = (title, text)
            self._tf[url] = counts
            self._length[url] = len(tokens)
        self._df: dict[str, int] = {}
        for counts in self._tf.values():
            for token in counts:
                self._df[token] = self._df.get(token, 0) + 1
        self._avg_length = (sum(self._length.values()) / len(self._length)) if self._length else 1.0

    def __len__(self) -> int:
        return len(self.pages)

    def rank(self, query: dict[str, float], allowed: set[str]) -> list[tuple[float, str]]:
        total = len(self.pages)
        scored: list[tuple[float, str]] = []
        for url in allowed:
            counts, length = self._tf[url], self._length[url]
            score = 0.0
            for token, weight in query.items():
                tf = counts.get(token)
                if not tf:
                    continue
                df = self._df[token]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / self._avg_length)
                score += weight * idf * tf * (BM25_K1 + 1) / norm
            if score > 0:
                scored.append((round(score, 3), url))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored


def load_docs_index() -> DocsIndex:
    index = DocsIndex(fetch_text(LLMS_FULL_URL))
    if len(index) == 0:
        raise RuntimeError(f"{LLMS_FULL_URL} contained no pages in the expected format")
    return index


def issue_query_terms(title: str, body: str) -> dict[str, float]:
    """BM25 query: every term of the title and the kept, noise-stripped body sections,
    weighted by how often the issue uses it (standard query term frequency)."""
    terms: dict[str, float] = {}
    for token in tokenize(f"{title}\n{relevant_body_text(body)}"):
        terms[token] = terms.get(token, 0.0) + 1.0
    return terms


def candidate_urls(index: DocsIndex, haystack: str) -> set[str]:
    """Every indexed page except release notes and, unless named, the language ports."""
    allowed = set()
    for url in index.pages:
        if EXCLUDED_PAGE_RE.match(url):
            continue
        port = next(
            (name for name, prefix in PORT_PAGE_PREFIXES.items() if url.startswith(prefix)), None
        )
        if port and port not in haystack:
            continue
        allowed.add(url)
    return allowed


def pick_pages(
    index: DocsIndex, ranked: list[tuple[float, str]], issue_doc_urls: list[str]
) -> list[str]:
    """Up to MAX_PAGES page URLs: pages the issue links first, then the best-ranked ones."""
    picked: list[str] = []
    by_key = {normalize_docs_url(url): url for url in index.pages}

    def add(url: str) -> None:
        if url not in picked and len(picked) < MAX_PAGES:
            picked.append(url)

    for url in issue_doc_urls:
        known = by_key.get(normalize_docs_url(url))
        if known:
            add(known)
    top_score = ranked[0][0] if ranked else 0.0
    for score, url in ranked:
        if score < top_score * MIN_RELATIVE_SCORE:
            break
        add(url)
    return picked


def read_tool_prompt(prompt_name: str) -> str:
    return (Path(__file__).parent / "prompts" / prompt_name).read_text(encoding="utf-8")


def build_site_check_user_message(
    row: dict[str, Any], candidates: list[tuple[str, str]], pages: dict[str, str]
) -> str:
    body = row["_body"][:ISSUE_BODY_MAX_CHARS]
    parts = [
        f"GitHub issue #{row['number']}: {row['title']}",
        "",
        "Issue body:",
        body or "(empty)",
        "",
        "Candidate pages, best lexical match first (title: url):",
        *(f"- {title}: {url}" for title, url in candidates),
        "",
    ]
    if pages:
        parts.append("Fetched pages:")
        for url, text in pages.items():
            parts.extend(["", f"=== {url}", "", text])
    else:
        parts.append("No page bodies were fetched; only the candidate list above is available.")
    return "\n".join(parts)


async def site_check_with_llm(system_prompt: str, user_message: str) -> Any:
    """One structured LLM call. Same instructor/litellm block as tools/assess_branch_notes.py."""
    try:
        import instructor
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(f"Required dependencies not available: {exc}") from exc

    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set")

    from typing import Literal

    class CoveringPage(BaseModel):
        url: str = Field(description="URL of a provided page, exactly as listed")
        note: str = Field(
            description=(
                "One sentence, no more, saying what this page states that resolves the report. "
                "Starts with a verb, e.g. 'states that TRIPLET_COMPLETION requires ...'"
            )
        )

    class SiteCheck(BaseModel):
        verdict: Literal["not_docs", "too_vague", "documentation_covered", "needs_source"]
        reason: str = Field(description="One or two sentences justifying the verdict")
        covering_pages: list[CoveringPage] = Field(
            default_factory=list,
            description="Provided pages that resolve the report (documentation_covered only)",
        )

    try:
        client = instructor.from_litellm(litellm.acompletion)
        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_model=SiteCheck,
            api_key=api_key,
            max_retries=2,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM site check failed: {exc}") from exc


def run_site_check(system_prompt: str, user_message: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Sync wrapper around the LLM call: (verdict, reason, [(url, note), ...]).

    Tests replace this function.
    """
    result = asyncio.run(site_check_with_llm(system_prompt, user_message))
    return result.verdict, result.reason, [(page.url, page.note) for page in result.covering_pages]


def check_issue_against_docs(row: dict[str, Any], index: DocsIndex) -> None:
    """Fill ``verdict`` / ``reason`` / ``doc_urls`` on a ``pending_docs_check`` row."""
    body = row["_body"]
    haystack = f"{row['title']}\n{relevant_body_text(body)}".lower()
    ranked = index.rank(issue_query_terms(row["title"], body), candidate_urls(index, haystack))
    page_urls = pick_pages(index, ranked, docs_site_urls(body))
    pages = {url: index.pages[url][1][:PAGE_MAX_CHARS] for url in page_urls}
    listed = [(index.pages[url][0], url) for url in page_urls]
    listed += [
        (index.pages[url][0], url)
        for _score, url in ranked[:MAX_LISTED_CANDIDATES]
        if url not in page_urls
    ][: max(0, MAX_LISTED_CANDIDATES - len(listed))]

    verdict, reason, covering = run_site_check(
        read_tool_prompt(SITE_CHECK_PROMPT),
        build_site_check_user_message(row, listed, pages),
    )
    if verdict not in SITE_CHECK_VERDICTS:
        verdict, reason = "needs_source", f"unexpected verdict {verdict!r}: {reason}"

    # Keep only pages the LLM was actually shown, with the index's title for the link text.
    fetched = {normalize_docs_url(url): url for url in pages}
    doc_pages: list[dict[str, str]] = []
    for url, note in covering:
        known = fetched.get(normalize_docs_url(url))
        if known and all(page["url"] != known for page in doc_pages):
            doc_pages.append({"title": index.pages[known][0], "url": known, "note": note.strip()})
    if verdict == "documentation_covered" and not doc_pages:
        verdict = "needs_source"
        reason = f"{reason} (downgraded: cited pages were not among those provided)"

    row["verdict"] = verdict
    row["reason"] = reason
    row["pages_shown"] = list(pages)
    row["doc_pages"] = doc_pages if verdict == "documentation_covered" else []
    row["doc_urls"] = [page["url"] for page in row["doc_pages"]]


def issue_comments(repo: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    """The issue's comments, fetched once per row and reused by every later check."""
    if "_comments" not in row:
        row["_comments"] = list_issue_comments(repo, row["number"])
    return row["_comments"]


def is_bot_comment(comment: dict[str, Any]) -> bool:
    user = comment.get("user") or {}
    login = (user.get("login") or "").lower()
    return user.get("type") == "Bot" or login.endswith("[bot]") or login == "github-actions"


def maintainer_reply_reason(comments: list[dict[str, Any]], core_team: set[str]) -> str | None:
    """Why the thread is already in a maintainer's hands, or None."""
    for comment in comments:
        body = comment.get("body") or ""
        if is_bot_comment(comment) or COMMENT_MARKER in body.splitlines():
            continue
        if is_maintainer(comment, core_team):
            login = (comment.get("user") or {}).get("login") or "a maintainer"
            day = (comment.get("created_at") or "")[:10]
            return f"maintainer @{login} replied{' on ' + day if day else ''}; left to them"
    return None


def is_bug_report(title: str, labels: list[str]) -> bool:
    """Filed through the bug form: ``[Bug]:`` title prefix or the ``bug`` label."""
    return bool(BUG_TITLE_RE.search(title or "")) or any(
        name.lower() == BUG_LABEL for name in labels
    )


def list_issue_comments(repo: str, number: int) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"per_page": "100", "page": str(page)})
        batch = github_api_json(f"{GITHUB_API}/repos/{repo}/issues/{number}/comments?{query}")
        if not batch:
            return comments
        comments.extend(batch)
        page += 1


def has_triage_comment(comments: list[dict[str, Any]]) -> bool:
    return any(COMMENT_MARKER in (comment.get("body") or "").splitlines() for comment in comments)


def documentation_covered_comment(doc_pages: list[dict[str, str]]) -> str:
    links = "\n".join(
        f"- [{page['title']}]({page['url']})" + (f": {page['note']}" if page["note"] else "")
        for page in doc_pages
    )
    return (
        f"{COMMENT_MARKER}\n"
        "This comment is auto-generated.\n\n"
        "The public Cognee docs already cover this report:\n\n"
        f"{links}\n\n"
        "If that answers your question, please close this issue. "
        "A maintainer will not auto-close it.\n"
    )


def post_issue_comment(repo: str, number: int, body: str) -> None:
    github_api_json(
        f"{GITHUB_API}/repos/{repo}/issues/{number}/comments", method="POST", payload={"body": body}
    )


def maybe_comment_documentation_covered(repo: str, row: dict[str, Any], dry_run: bool) -> None:
    if row["verdict"] != "documentation_covered":
        return
    row["comment"] = documentation_covered_comment(row["doc_pages"])
    if dry_run:
        row["reason"] = f"{row['reason']} (dry run: comment suppressed)"
        return
    if has_triage_comment(issue_comments(repo, row)):
        row["reason"] = f"{row['reason']} (bot comment already present)"
        return
    post_issue_comment(repo, row["number"], row["comment"])
    row["commented"] = True


def run_docs_check(
    repo: str, results: list[dict[str, Any]], dry_run: bool, repo_root: Path
) -> tuple[bool, DocsIndex | None]:
    """Phase 2 over every pending row: (all LLM calls succeeded, the docs index or None)."""
    pending = [row for row in results if row["verdict"] == "pending_docs_check"]
    if not pending:
        return True, None
    # Two decisions that need no LLM: a maintainer who already replied owns the thread,
    # and a bug report asks for a code change, not a docs edit.
    core_team = core_team_logins(repo_root)
    for row in pending:
        replied = maintainer_reply_reason(issue_comments(repo, row), core_team)
        if replied:
            row["verdict"] = "maintainer_replied"
            row["reason"] = replied
        elif is_bug_report(row["title"], row["_labels"]):
            row["verdict"] = "not_docs"
            row["reason"] = "filed through the bug form: a code change, not a docs edit"
    pending = [row for row in pending if row["verdict"] == "pending_docs_check"]
    if not pending:
        return True, None
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError(
            f"{len(pending)} issue(s) passed the filter but LLM_API_KEY is not set; "
            "set it, or select only issues that the cheap filter skips."
        )
    index = load_docs_index()  # one fetch per run, shared by every pending issue
    ok = True
    for row in pending:
        try:
            check_issue_against_docs(row, index)
        except RuntimeError as exc:
            row["verdict"] = "uncertain"
            row["reason"] = f"docs check failed: {exc}"
            ok = False
            continue
        maybe_comment_documentation_covered(repo, row, dry_run)
    return ok, index


# --- phase 3: source check, human review, draft-docs hand-off ---------------------------------


def issue_identifiers(title: str, body: str) -> list[str]:
    """Code-like tokens the issue names, longest first, deduplicated, capped."""
    text = f"{title}\n{body}"
    found: list[str] = []
    for pattern in IDENTIFIER_RES:
        for match in pattern.findall(text):
            token = match.strip().strip("`'\"()[]{},.;:")
            if len(token) >= GREP_MIN_TOKEN_LEN and " " not in token and token not in found:
                found.append(token)
    found.sort(key=len, reverse=True)
    return found[:GREP_MAX_TOKENS]


def git_grep_hits(tokens: list[str], repo_root: Path) -> str:
    """One fixed-string git grep over the source for every token; empty when nothing matches."""
    if not tokens:
        return ""
    command = ["git", "-C", str(repo_root), "grep", "-I", "-n", "-F"]
    for token in tokens:
        command += ["-e", token]
    command += ["--", *GREP_PATHS, *GREP_EXCLUDES]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode not in (0, 1):  # 1 = no match
        raise RuntimeError(f"git grep failed ({completed.returncode}): {completed.stderr.strip()}")
    return completed.stdout[:GREP_MAX_CHARS]


def linked_pull_requests(repo: str, number: int, core_team: set[str]) -> list[str]:
    """OPEN pull requests by a MAINTAINER that reference the issue, as "url by @login".

    A closed or merged PR is not a fix in progress: either the change landed, and the
    source check will see it, or it was abandoned. A contributor's PR may never merge,
    so it does not count either; the timeline carries the PR author's association.
    """
    urls: list[str] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"per_page": "100", "page": str(page)})
        batch = github_api_json(f"{GITHUB_API}/repos/{repo}/issues/{number}/timeline?{query}")
        if not batch:
            return urls
        for event in batch:
            if event.get("event") != "cross-referenced":
                continue
            source_issue = (event.get("source") or {}).get("issue") or {}
            if (
                "pull_request" in source_issue
                and source_issue.get("state") == "open"
                and source_issue.get("html_url")
                and is_maintainer(source_issue, core_team)
            ):
                login = (source_issue.get("user") or {}).get("login") or "a maintainer"
                entry = f"{source_issue['html_url']} by @{login}"
                if entry not in urls:
                    urls.append(entry)
        page += 1


def core_team_logins(repo_root: Path) -> set[str]:
    """Logins from .github/core-team.txt (one per line, optional @, '#' comments)."""
    path = repo_root / CORE_TEAM_FILE
    if not path.is_file():
        return set()
    logins = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            logins.add(line.lstrip("@").lower())
    return logins


def is_maintainer(comment: dict[str, Any], core_team: set[str]) -> bool:
    login = ((comment.get("user") or {}).get("login") or "").lower()
    return comment.get("author_association") in MAINTAINER_ASSOCIATIONS or (
        bool(login) and login in core_team
    )


def fix_in_progress_reason(repo: str, row: dict[str, Any], repo_root: Path) -> str | None:
    """Why the issue is already being fixed, or None: an open PR by a maintainer.

    Volunteer comments are not checked here: a maintainer's reply already ends the
    triage as maintainer_replied in phase 2, and anyone else's promise is not a fix.
    """
    pull_requests = linked_pull_requests(repo, row["number"], core_team_logins(repo_root))
    if pull_requests:
        return f"open maintainer pull request: {', '.join(pull_requests[:3])}"
    return None


def build_source_check_user_message(
    row: dict[str, Any], docs_pages: dict[str, str], grep_hits: str
) -> str:
    parts = [
        f"GitHub issue #{row['number']}: {row['title']}",
        "",
        "Issue body:",
        row["_body"][:ISSUE_BODY_MAX_CHARS] or "(empty)",
        "",
        "Docs pages already checked (they did not answer the report):",
    ]
    for url, text in docs_pages.items():
        parts += ["", f"=== {url}", "", text]
    parts += [
        "",
        "git grep hits in the cognee source (path:line:text):",
        grep_hits or "(no hits for the identifiers named in the issue)",
    ]
    return "\n".join(parts)


async def source_check_with_llm(system_prompt: str, user_message: str) -> Any:
    """Second structured LLM call, same instructor/litellm block as the site check."""
    try:
        import instructor
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(f"Required dependencies not available: {exc}") from exc

    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set")

    from typing import Literal

    class SourceCheck(BaseModel):
        verdict: Literal["not_in_source", "too_big", "uncertain", "small_gap"]
        reason: str = Field(description="One or two sentences justifying the verdict")
        source_files: list[str] = Field(
            default_factory=list,
            description="cognee/ or .env.template paths from the grep hits that prove the behaviour",
        )
        docs_files: list[str] = Field(
            default_factory=list,
            description="At most 3 existing docs paths relative to the docs repo, e.g. guides/x.mdx",
        )

    try:
        client = instructor.from_litellm(litellm.acompletion)
        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_model=SourceCheck,
            api_key=api_key,
            max_retries=2,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM source check failed: {exc}") from exc


def run_source_check(
    system_prompt: str, user_message: str
) -> tuple[str, str, list[str], list[str]]:
    """Sync wrapper: (verdict, reason, source_files, docs_files). Tests replace this."""
    result = asyncio.run(source_check_with_llm(system_prompt, user_message))
    return result.verdict, result.reason, list(result.source_files), list(result.docs_files)


def _clean_docs_path(path: str) -> str | None:
    path = path.strip().strip("`").lstrip("./")
    if not path or path.startswith(("http://", "https://")) or " " in path:
        return None
    return path


def check_issue_against_source(row: dict[str, Any], index: DocsIndex, repo_root: Path) -> None:
    """Fill verdict / reason / source_files / docs_files on a ``needs_source`` row."""
    grep_hits = git_grep_hits(issue_identifiers(row["title"], row["_body"]), repo_root)
    docs_pages = {url: index.pages[url][1][:PAGE_MAX_CHARS] for url in row["pages_shown"]}
    verdict, reason, source_files, docs_files = run_source_check(
        read_tool_prompt(SOURCE_CHECK_PROMPT),
        build_source_check_user_message(row, docs_pages, grep_hits),
    )
    if verdict not in SOURCE_CHECK_VERDICTS:
        verdict, reason = "uncertain", f"unexpected verdict {verdict!r}: {reason}"

    cleaned_docs = [p for p in (_clean_docs_path(f) for f in docs_files) if p][:MAX_DOCS_FILES]
    if verdict == "small_gap" and not cleaned_docs:
        verdict = "uncertain"
        reason = f"{reason} (downgraded: no existing docs file named)"

    row["verdict"] = verdict
    row["reason"] = reason
    row["source_files"] = (
        [f.strip() for f in source_files if f.strip()] if verdict == "small_gap" else []
    )
    row["docs_files"] = cleaned_docs if verdict == "small_gap" else []


def run_source_checks(
    repo: str, results: list[dict[str, Any]], index: DocsIndex | None, repo_root: Path
) -> bool:
    """Phase 3 over every ``needs_source`` row. Returns False if any LLM call failed."""
    pending = [row for row in results if row["verdict"] == "needs_source"]
    if not pending:
        return True
    if index is None:
        index = load_docs_index()
    ok = True
    for row in pending:
        try:
            in_progress = fix_in_progress_reason(repo, row, repo_root)
            if in_progress:
                row["verdict"] = "fix_in_progress"
                row["reason"] = f"{row['reason']} (not drafting docs: {in_progress})"
                continue
            check_issue_against_source(row, index, repo_root)
        except RuntimeError as exc:
            row["verdict"] = "uncertain"
            row["reason"] = f"source check failed: {exc}"
            ok = False
    return ok


def human_review_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in results if row["verdict"] in HUMAN_REVIEW_VERDICTS]


def human_review_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Issue | Verdict | Reason |", "|---|---|---|"]
    for row in rows:
        lines.append(
            f"| #{row['number']} {_cell(row['title'])} | `{row['verdict']}` | {_cell(row['reason'])} |"
        )
    return lines


def run_url() -> str | None:
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


SMTP_ENV = (
    "SMTP_SERVER",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "NOTIFICATION_EMAIL_SENDER",
    "NOTIFICATION_EMAIL_RECEIVER",
)


def send_human_review_email(rows: list[dict[str, Any]], dry_run: bool) -> str:
    """Email the human-review list. Returns a one-line status for the summary.

    Same SMTP env as the notify-failure job of dev_previous_day_commits.yml; when any
    of it is missing the email is skipped and that is not an error.
    """
    if not rows:
        return "no items need human review"
    missing = [name for name in SMTP_ENV if not os.environ.get(name)]
    if missing:
        return f"{len(rows)} item(s) need human review; email skipped, SMTP not configured ({', '.join(missing)})"

    subject = f"docs issue triage: {len(rows)} item(s) need human review"
    header = textwrap.dedent(
        f"""\
        {len(rows)} GitHub issue(s) passed the docs triage but need a human decision.
        {"This was a DRY RUN: no comment and no docs PR was made." if dry_run else ""}
        """
    ).strip()
    url = run_url()
    body_lines = [header, "", f"Run: {url}" if url else "", ""]
    body_lines += human_review_table_lines(rows)
    body_lines += ["", "Issue links:"] + [f"- {row['html_url']}" for row in rows]

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = os.environ["NOTIFICATION_EMAIL_SENDER"]
    message["To"] = os.environ["NOTIFICATION_EMAIL_RECEIVER"]
    message.set_content("\n".join(line for line in body_lines if line is not None))

    smtp_port = int(os.environ.get("SMTP_PORT") or "587")
    use_tls = (os.environ.get("SMTP_USE_TLS") or "true").lower() in {"1", "true", "yes"}
    with smtplib.SMTP(os.environ["SMTP_SERVER"], smtp_port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(message)
    return (
        f"emailed {len(rows)} human-review item(s) to {os.environ['NOTIFICATION_EMAIL_RECEIVER']}"
    )


def matrix_rows(results: list[dict[str, Any]], dry_run: bool) -> list[dict[str, str]]:
    """Flat-string rows for the draft-docs matrix; empty on dry runs."""
    if dry_run:
        return []
    rows: list[dict[str, str]] = []
    for row in results:
        if row["verdict"] not in GAP_VERDICTS:
            continue
        excerpt = f"{row['title']}\n\n{row['_body']}".encode()[:MATRIX_EXCERPT_MAX_BYTES]
        rows.append(
            {
                "number": str(row["number"]),
                "title": row["title"][:120],
                "body_b64": base64.b64encode(excerpt).decode("ascii"),
                "source_files": " ".join(f for f in row["source_files"] if " " not in f),
                "docs_files": " ".join(f for f in row["docs_files"] if " " not in f),
            }
        )
    return rows


def post_gap_comment(repo: str, number: int, dry_run: bool) -> str:
    """The phase 3 defensive comment, once per issue. Returns a status line."""
    if dry_run:
        return f"dry run: gap comment on #{number} not posted"
    if has_triage_comment(list_issue_comments(repo, number)):
        return f"#{number} already carries the bot comment; nothing posted"
    post_issue_comment(repo, number, GAP_COMMENT)
    return f"posted the gap comment on #{number}"


def public_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in results]


def write_step_summary(
    args: argparse.Namespace, results: list[dict[str, Any]], email_status: str = ""
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "## Docs issue triage",
        "",
        f"- Repository: `{args.repo}`",
        f"- Selector: {describe_selector(args)}",
        f"- Dry run: `{'true' if args.dry_run else 'false'}`",
        *summary_count_lines(results),
        "",
        (
            "Public actions: one marked comment on `documentation_covered` issues, and for "
            "`small_gap` issues a draft PR on topoteretes/cognee-docs plus one defensive "
            "comment (both from the draft-docs job). All suppressed on dry run. Never closes."
        ),
        "",
        *summary_table_lines(results),
        "",
    ]
    review = human_review_rows(results)
    if review:
        lines += ["### Human review", "", *human_review_table_lines(review), ""]
    if email_status:
        lines += [f"- Email: {email_status}", ""]
    with Path(summary_path).open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_github_output(results: list[dict[str, Any]], dry_run: bool) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    rows = matrix_rows(results, dry_run)
    with Path(github_output).open("a", encoding="utf-8") as fh:
        fh.write(f"has_gaps={'true' if rows else 'false'}\n")
        fh.write(f"matrix={json.dumps(rows)}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.post_gap_comment is not None:
        try:
            print(post_gap_comment(args.repo, args.post_gap_comment, args.dry_run))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"GitHub API request failed: {exc.code} {exc.reason}\n{detail}", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"GitHub API request failed: {exc.reason}", file=sys.stderr)
            return 1
        return 0

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

    exit_code = 0
    email_status = ""
    try:
        docs_ok, index = run_docs_check(args.repo, results, args.dry_run, args.repo_root)
        source_ok = run_source_checks(args.repo, results, index, args.repo_root)
        if not (docs_ok and source_ok):
            exit_code = 1
        email_status = send_human_review_email(human_review_rows(results), args.dry_run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        exit_code = 1
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP request failed: {exc.code} {exc.reason} {exc.url}\n{detail}", file=sys.stderr)
        exit_code = 1
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"HTTP request failed: {exc}", file=sys.stderr)
        exit_code = 1
    except (smtplib.SMTPException, OSError) as exc:
        email_status = f"email failed: {exc}"
        print(email_status, file=sys.stderr)
        exit_code = 1

    rows = public_rows(results)
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print(f"Selector: {describe_selector(args)}")
    print("\n".join(summary_count_lines(rows)))
    print("\n".join(summary_table_lines(rows)))
    if email_status:
        print(email_status)
    write_step_summary(args, rows, email_status)
    write_github_output(results, args.dry_run)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
