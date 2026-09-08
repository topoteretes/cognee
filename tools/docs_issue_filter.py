#!/usr/bin/env python3
"""Cheap, LLM-free filter: does a GitHub issue even look like a docs report?

Used by tools/docs_issue_triage.py as the first gate. Pure functions only, no I/O.
"""

from __future__ import annotations

import re
from typing import Any

# Whole words only, so "docstring" and "Docker" do not count.
DOCS_WORD_RE = re.compile(r"\bdocs\b|\bdocumentation\b", re.IGNORECASE)

DOCS_LABEL = "documentation"


def label_names(labels: list[Any] | None) -> list[str]:
    """Normalize GitHub label objects (dicts with a ``name`` key) or bare strings."""
    names: list[str] = []
    for label in labels or []:
        if isinstance(label, dict):
            name = label.get("name")
        else:
            name = label
        if isinstance(name, str):
            names.append(name)
    return names


def looks_like_docs_issue(labels: list[Any] | None, title: str | None, body: str | None) -> bool:
    """Return True when the issue carries the ``documentation`` label or a docs word.

    Checked in order: label, title, body. The ``needs-triage`` label the issue form
    applies is deliberately ignored; the repo does not really use it.
    """
    if any(name.lower() == DOCS_LABEL for name in label_names(labels)):
        return True
    if DOCS_WORD_RE.search(title or ""):
        return True
    return bool(DOCS_WORD_RE.search(body or ""))


def docs_match_reason(labels: list[Any] | None, title: str | None, body: str | None) -> str:
    """Short human-readable reason for the run summary, mirroring the filter order."""
    if any(name.lower() == DOCS_LABEL for name in label_names(labels)):
        return "label: documentation"
    if DOCS_WORD_RE.search(title or ""):
        return "title mentions docs"
    if DOCS_WORD_RE.search(body or ""):
        return "body mentions docs"
    return "no documentation label or docs word in title/body"
