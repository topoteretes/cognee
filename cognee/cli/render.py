"""Shared search/recall result rendering.

Contract: the answer goes to stdout (clean for pipes and files); provenance
and guidance go to stderr as a dim one-line footer. Previously this logic was
duplicated across search_command, recall_command, and api_dispatch.
"""

import json
import sys
from typing import Optional, Sequence

from cognee.cli import ui


def render_results(
    results: Sequence,
    query_type: str,
    output_format: str = "pretty",
    elapsed: Optional[float] = None,
    caps: Optional[ui.TermCaps] = None,
) -> None:
    caps = caps or ui.detect_caps()
    style = ui.Style(caps.color)

    if output_format == "json":
        sys.stdout.write(json.dumps(results, indent=2, default=str) + "\n")
        return

    if output_format == "simple":
        for index, result in enumerate(results, 1):
            sys.stdout.write(f"{index}. {result}\n")
        return

    # pretty
    if not results:
        sys.stderr.write(
            style.dim("No results for that query — try rephrasing, or check ")
            + style.dim("`cognee-cli datasets list` for what's in memory.")
            + "\n"
        )
        return

    completion_types = {
        "GRAPH_COMPLETION",
        "RAG_COMPLETION",
        "HYBRID_COMPLETION",
        "TRIPLET_COMPLETION",
        "GRAPH_COMPLETION_COT",
        "GRAPH_COMPLETION_CONTEXT_EXTENSION",
        "GRAPH_SUMMARY_COMPLETION",
        "GRAPH_COMPLETION_DECOMPOSITION",
        "NATURAL_LANGUAGE",
        "FEELING_LUCKY",
        "AGENTIC_COMPLETION",
    }

    if query_type in completion_types:
        for index, result in enumerate(results, 1):
            if index > 1:
                sys.stdout.write("\n")
            sys.stdout.write(f"{result}\n")
    elif query_type in ("CHUNKS", "CHUNKS_LEXICAL", "SUMMARIES"):
        for index, result in enumerate(results, 1):
            sys.stdout.write(f"{index}. {result}\n\n")
    else:
        for index, result in enumerate(results, 1):
            sys.stdout.write(f"{index}. {result}\n")

    footer_parts = [f"{len(results)} result{'s' if len(results) != 1 else ''}"]
    footer_parts.append(query_type.lower())
    if elapsed is not None:
        footer_parts.append(ui.format_duration(elapsed))
    glyphs = ui.Glyphs(caps.unicode)
    separator = f" {glyphs.sep} "
    sys.stderr.write(style.dim(f"— {separator.join(footer_parts)}") + "\n")


def render_session_entries(results: Sequence, caps: Optional[ui.TermCaps] = None) -> None:
    """Session-memory entries: [time] Q:/A: pairs (recall --session-id)."""
    caps = caps or ui.detect_caps()
    # Two style gates: stdout content must not carry ANSI into pipes even
    # when stderr is still a terminal (caps.color follows stderr).
    out_style = ui.Style(caps.color and caps.stdout_tty)
    err_style = ui.Style(caps.color)
    for index, entry in enumerate(results, 1):
        if index > 1:
            sys.stdout.write("\n")
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        stamp = entry.get("time", "")
        header = out_style.dim(f"[{stamp}] ") if stamp else ""
        if question:
            sys.stdout.write(f"{header}{out_style.bold('Q:')} {question}\n")
        if answer:
            sys.stdout.write(f"{out_style.bold('A:')} {answer}\n")
    footer = f"— {len(results)} session entr{'ies' if len(results) != 1 else 'y'}"
    sys.stderr.write(err_style.dim(footer) + "\n")
