#!/usr/bin/env python3
"""Cheap, LLM-free filter: does a GitHub issue even look like a docs report?

Used by tools/docs_issue_triage.py as the first gate. Pure functions only, no I/O.

The filter is signal-based rather than word-based. A bare "docs" anywhere in a
body is a poor predictor: a full-backlog dry run showed 42 of 44 hits came from
the body, almost all of them third-party ``/docs/`` links or "add docs" deliverables
in feature and hackathon templates. So the body is first reduced to the sections
where a user describes a problem, then stripped of code, links, quotes and
checklists, and only then searched for phrases that make a *claim about the
documentation* ("the docs say", "documented as", "not documented", "docs are
outdated"). A link to docs.cognee.ai is a positive signal in its own right.
"""

from __future__ import annotations

import re
from typing import Any

DOCS_LABEL = "documentation"

# Whole words only, so "docstring" and "Docker" do not count. Used on titles.
DOCS_WORD_RE = re.compile(r"\bdocs\b|\bdocumentation\b", re.IGNORECASE)

# ``[Docs]:`` / ``docs:`` / ``Documentation:`` title prefixes from the issue form
# and from conventional-commit habits.
DOCS_TITLE_PREFIX_RE = re.compile(r"^\s*(?:\[\s*docs?\s*\]|docs?|documentation)\s*:", re.IGNORECASE)

DOCS_SITE_URL_RE = re.compile(r"https?://docs\.cognee\.ai(?:/[^\s)>\]\"'`]*)?", re.IGNORECASE)

# Section headings GitHub issue forms render from the templates in
# .github/ISSUE_TEMPLATE/, plus the hackathon task family. Lower-cased.
DOCS_FORM_HEADINGS = frozenset(
    {"documentation type", "documentation location", "issue description", "suggested improvement"}
)
# Sections that hold logs, environment dumps, reproduction commands, deliverable
# lists or scaffolding, none of which describe a documentation problem.
DROPPED_HEADINGS = frozenset(
    {
        "environment",
        "logs/error messages",
        "logs",
        "steps to reproduce",
        "reproduction",
        "pre-submission checklist",
        "implementation ideas",
        "alternatives considered",
        "acceptance criteria",
        "acceptance",
        "watch out for",
        "package layout to follow",
        "version",
    }
)
DROPPED_HEADING_PREFIXES = ("build it on ",)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_AUTOLINK_RE = re.compile(r"<https?://[^>\s]+>")
_BARE_URL_RE = re.compile(r"https?://\S+")
_QUOTE_LINE_RE = re.compile(r"^\s{0,3}>.*$", re.MULTILINE)
_CHECKLIST_LINE_RE = re.compile(r"^\s*[-*+]\s+\[[ xX]\].*$", re.MULTILINE)
_NO_RESPONSE_RE = re.compile(r"_no response_", re.IGNORECASE)

# "the docs", "our documentation", "the docs page", "cognee's documentation site", ...
_DOCS_NOUN = (
    r"(?:(?:the|our|your|their|its|cognee'?s?|official|public|current|existing|online|"
    r"search-types?|config|prune|setup|api|python|main|linked|relevant)\s+){0,3}"
    r"(?:docs?|documentation)(?:\s+(?:page|pages|site|section|entry|table|reference))?"
)
_SAY_VERBS = (
    r"say|says|said|state|states|stated|claim|claims|claimed|list|lists|listed|describe|"
    r"describes|described|mention|mentions|mentioned|promise|promises|promised|document|"
    r"documents|show|shows|showed|read|reads|suggest|suggests|imply|implies|tell|tells|"
    r"recommend|recommends|acknowledge|acknowledges|present|presents|advise|advises|"
    r"instruct|instructs|cover|covers|explain|explains|warn|warns|call|calls|refer|refers|"
    r"define|defines|specify|specifies|indicate|indicates"
)
_BAD_ADJ = (
    r"unclear|wrong|incorrect|outdated|out\s+of\s+date|missing|misleading|incomplete|stale|"
    r"confusing|inaccurate|inconsistent|silent|ambiguous|contradictory|broken"
)
_ADVERBS = r"(?:(?:already|currently|still|explicitly|clearly|only|also|now|even|simply)\s+)*"

DOCS_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # "the docs currently state", "documentation lists", "the search-type docs describe"
        rf"\b{_DOCS_NOUN}\s+{_ADVERBS}(?:{_SAY_VERBS})\b",
        # "the docs do not mention", "documentation never warns"
        (
            rf"\b{_DOCS_NOUN}\s+(?:do(?:es)?\s+not|don'?t|doesn'?t|never|no\s+longer|nowhere)\s+"
            rf"(?:{_SAY_VERBS}|include|point|hint|address)"
        ),
        # "the documentation is unclear", "docs are out of date"
        (
            rf"\b{_DOCS_NOUN}\s+(?:is|are|was|were|seems?|looks?|remains?|reads?\s+as)\s+"
            rf"(?:\w+\s+){{0,2}}?(?:{_BAD_ADJ})\b"
        ),
        # "unclear documentation", "outdated docs"
        rf"\b(?:{_BAD_ADJ})\s+{_DOCS_NOUN}\b",
        # "is documented as setting", "was documented to do X" -- a present/past-tense
        # statement about what the docs say, not "should be documented in the README"
        r"\b(?:is|are|was|were|been|gets?|currently|explicitly)\s+documented\s+(?:as|to|that)\b",
        # "undocumented", "not documented", "never documented", "nowhere documented"
        r"\b(?:un|not\s+|never\s+|nowhere\s+|isn'?t\s+|aren'?t\s+)documented\b",
        # "nothing in the docs", "missing from the documentation", "not on the docs site"
        rf"\b(?:nothing|nowhere|missing|absent|not|anywhere)\s+(?:in|on|from)\s+{_DOCS_NOUN}\b",
        # "according to the docs", "per the documentation", "contrary to the docs"
        rf"\b(?:according\s+to|per|as\s+per|contrary\s+to|based\s+on|following)\s+{_DOCS_NOUN}\b",
        # "docs gap", "documentation bug", "docs typo"
        r"\b(?:docs?|documentation)\s+(?:gap|bug|error|typo|mismatch|discrepancy|issue)\b",
        r"\btypo\s+in\s+(?:the\s+)?(?:docs?|documentation|page)\b",
        r"\bbroken\s+(?:docs?\s+)?link\b",
    )
)


def label_names(labels: list[Any] | None) -> list[str]:
    """Normalize GitHub label objects (dicts with a ``name`` key) or bare strings."""
    names: list[str] = []
    for label in labels or []:
        name = label.get("name") if isinstance(label, dict) else label
        if isinstance(name, str):
            names.append(name)
    return names


def split_sections(body: str) -> list[tuple[str | None, str]]:
    """Split a body into ``(heading, text)`` pairs on Markdown headings.

    Text before the first heading gets ``None`` as its heading. Heading text is kept
    inside the section text, since free-form headings ("What the documentation
    promises") are user prose too.
    """
    sections: list[tuple[str | None, str]] = []
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [(None, body)]
    if matches[0].start() > 0:
        sections.append((None, body[: matches[0].start()]))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        heading = match.group(1).strip()
        sections.append((heading, body[match.start() : end]))
    return sections


def is_dropped_heading(heading: str | None) -> bool:
    if heading is None:
        return False
    normalized = heading.strip().strip(":").lower()
    if normalized in DROPPED_HEADINGS:
        return True
    return normalized.startswith(DROPPED_HEADING_PREFIXES)


def strip_noise(text: str) -> str:
    """Remove code, links, quotes, checklists and form placeholders from prose."""
    text = _FENCED_CODE_RE.sub(" ", text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _AUTOLINK_RE.sub(" ", text)
    text = _BARE_URL_RE.sub(" ", text)
    text = _QUOTE_LINE_RE.sub(" ", text)
    text = _CHECKLIST_LINE_RE.sub(" ", text)
    text = _NO_RESPONSE_RE.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text)


def relevant_body_text(body: str | None) -> str:
    """The prose worth searching: kept sections, stripped of noise."""
    if not body:
        return ""
    kept = [text for heading, text in split_sections(body) if not is_dropped_heading(heading)]
    return strip_noise("\n".join(kept))


def form_headings(body: str | None) -> set[str]:
    return {h.strip().strip(":").lower() for h, _ in split_sections(body or "") if h}


def docs_site_urls(body: str | None) -> list[str]:
    """Distinct docs.cognee.ai links in the raw body, in order of appearance."""
    seen: list[str] = []
    for url in DOCS_SITE_URL_RE.findall(body or ""):
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.append(url)
    return seen


def find_docs_claim(text: str) -> str | None:
    """The first phrase that makes a claim about the documentation, if any."""
    best: re.Match[str] | None = None
    for pattern in DOCS_CLAIM_PATTERNS:
        match = pattern.search(text)
        if match and (best is None or match.start() < best.start()):
            best = match
    if best is None:
        return None
    return re.sub(r"\s+", " ", best.group(0)).strip()


def docs_signals(labels: list[Any] | None, title: str | None, body: str | None) -> list[str]:
    """Every reason this issue looks like a docs report, strongest first.

    Empty list means the cheap filter says no. Order: label, title, documentation
    issue form, docs.cognee.ai link, docs-claim phrase in the relevant prose.
    """
    signals: list[str] = []
    if any(name.lower() == DOCS_LABEL for name in label_names(labels)):
        signals.append("label: documentation")

    title = title or ""
    if DOCS_TITLE_PREFIX_RE.search(title):
        signals.append("title prefix")
    elif DOCS_WORD_RE.search(title):
        signals.append("title mentions docs")

    headings = form_headings(body)
    if len(headings & DOCS_FORM_HEADINGS) >= 2:
        signals.append("documentation issue form")

    urls = docs_site_urls(body)
    if urls:
        signals.append(f"links {urls[0]}" + (f" (+{len(urls) - 1})" if len(urls) > 1 else ""))

    claim = find_docs_claim(relevant_body_text(body))
    if claim:
        signals.append(f'claim: "{claim}"')

    return signals


def looks_like_docs_issue(labels: list[Any] | None, title: str | None, body: str | None) -> bool:
    """True when at least one docs signal fires. See ``docs_signals``."""
    return bool(docs_signals(labels, title, body))


def docs_match_reason(labels: list[Any] | None, title: str | None, body: str | None) -> str:
    """Short human-readable reason for the run summary."""
    signals = docs_signals(labels, title, body)
    return "; ".join(signals) if signals else "no documentation signal"
