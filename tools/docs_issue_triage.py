#!/usr/bin/env python3
"""Triage GitHub issues on topoteretes/cognee for documentation problems.

Phase 1 selects open issues by number or by created-at date range and runs the
cheap docs filter from tools/docs_issue_filter.py. Phase 2 (this version) takes
every issue that passed the filter, ranks the public docs pages against it with a
lexical (BM25) index built once per run from the site's full-text export, hands
the top pages to one structured LLM call that decides whether the site already
answers the report, and on ``already_answered`` posts a single marked comment
asking the author to close. Every other verdict stays silent. It never labels or closes an
issue. Later phases add the source check, the human-review email, and the
cognee-docs draft PR.

Exit codes: 2 bad arguments, 1 GitHub/HTTP/LLM failure, 0 success.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docs_issue_filter import docs_signals, docs_site_urls, relevant_body_text

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
SITE_CHECK_VERDICTS = ("not_docs", "too_vague", "already_answered", "needs_source")

COMMENT_MARKER = "<!-- cognee-docs-issue-triage -->"


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
        "doc_urls": [],
        "source_files": [],
        "docs_files": [],
        "commented": False,
        # Not part of the published row; carried so phase 2 can read the body once.
        "_body": issue.get("body") or "",
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
        "| Issue | Signals | Verdict | Reason | Pages shown to the LLM | Commented |",
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
        commented = "yes" if row["commented"] else "no"
        lines.append(
            f"| {link} {title} | {signals} | `{row['verdict']}` | {reason} | {pages} | {commented} |"
        )
    if not listed:
        lines.append("| _no issue passed the cheap filter_ | | | | | |")
    return lines


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


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

    class SiteCheck(BaseModel):
        verdict: Literal["not_docs", "too_vague", "already_answered", "needs_source"]
        reason: str = Field(description="One or two sentences justifying the verdict")
        doc_urls: list[str] = Field(
            default_factory=list,
            description="Fetched page URLs that already state the answer (already_answered only)",
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


def run_site_check(system_prompt: str, user_message: str) -> tuple[str, str, list[str]]:
    """Sync wrapper around the LLM call; tests replace this function."""
    result = asyncio.run(site_check_with_llm(system_prompt, user_message))
    return result.verdict, result.reason, list(result.doc_urls)


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

    verdict, reason, llm_urls = run_site_check(
        read_tool_prompt(SITE_CHECK_PROMPT),
        build_site_check_user_message(row, listed, pages),
    )
    if verdict not in SITE_CHECK_VERDICTS:
        verdict, reason = "needs_source", f"unexpected verdict {verdict!r}: {reason}"

    fetched = {normalize_docs_url(url): url for url in pages}
    confirmed: list[str] = []
    for url in llm_urls:
        key = normalize_docs_url(url)
        if key in fetched and fetched[key] not in confirmed:
            confirmed.append(fetched[key])
    if verdict == "already_answered" and not confirmed:
        verdict = "needs_source"
        reason = f"{reason} (downgraded: cited pages were not among those provided)"

    row["verdict"] = verdict
    row["reason"] = reason
    row["pages_shown"] = list(pages)
    row["doc_urls"] = confirmed if verdict == "already_answered" else []


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


def already_answered_comment(doc_urls: list[str]) -> str:
    links = "\n".join(f"- {url}" for url in doc_urls)
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


def maybe_comment_already_answered(repo: str, row: dict[str, Any], dry_run: bool) -> None:
    if row["verdict"] != "already_answered":
        return
    if dry_run:
        row["reason"] = f"{row['reason']} (dry run: comment suppressed)"
        return
    if has_triage_comment(list_issue_comments(repo, row["number"])):
        row["reason"] = f"{row['reason']} (bot comment already present)"
        return
    post_issue_comment(repo, row["number"], already_answered_comment(row["doc_urls"]))
    row["commented"] = True


def run_docs_check(repo: str, results: list[dict[str, Any]], dry_run: bool) -> bool:
    """Phase 2 over every pending row. Returns False if any LLM call failed."""
    pending = [row for row in results if row["verdict"] == "pending_docs_check"]
    if not pending:
        return True
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
        maybe_comment_already_answered(repo, row, dry_run)
    return ok


def public_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in results]


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
        *summary_count_lines(results),
        "",
        (
            "Phase 2: the only public action is one marked comment on `already_answered` "
            "issues (suppressed on dry run). No docs PRs, no email, no closing."
        ),
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

    exit_code = 0
    try:
        if not run_docs_check(args.repo, results, args.dry_run):
            exit_code = 1
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

    rows = public_rows(results)
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print(f"Selector: {describe_selector(args)}")
    print("\n".join(summary_count_lines(rows)))
    print("\n".join(summary_table_lines(rows)))
    write_step_summary(args, rows)
    write_github_output(rows, args.dry_run)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
