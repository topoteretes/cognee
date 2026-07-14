"""Citation formatter for the Support-Triage Bot.

Transforms cognee recall results into human-readable, numbered citations
with links and resolution summaries.
"""

from __future__ import annotations

from models import Citation, TriageResult


def format_triage_result(result: TriageResult) -> str:
    """Format a TriageResult into a user-facing message string.

    When citations are found:
        💡 Similar past issues found:
        [1] ... (resolution + link)
        [2] ...
        Suggested fix: ...

    When no citations:
        🔍 No similar resolved support threads were found.

    Args:
        result: The triage result to format.

    Returns:
        Formatted message string.
    """
    if not result.citations:
        return _format_no_citations()
    return _format_with_citations(result)


def _format_with_citations(result: TriageResult) -> str:
    """Format a triage result that has citations."""
    lines: list[str] = ["💡 **Similar past issues found:**", ""]

    for i, citation in enumerate(result.citations, 1):
        lines.append(_format_single_citation(i, citation))

    if result.suggested_reply:
        lines.append("")
        lines.append(f"**Suggested fix:** {result.suggested_reply}")

    lines.append("")
    lines.append("React ✅ to save this thread | Type `!forget` to remove from memory")

    return "\n".join(lines)


def _format_no_citations() -> str:
    """Format the fallback message when no similar issues are found."""
    return (
        "🔍 **No similar resolved support threads were found.**\n"
        "\n"
        "This appears to be a new problem. "
        "Once resolved, react ✅ to save it for future reference."
    )


def _format_single_citation(index: int, citation: Citation) -> str:
    """Format a single numbered citation.

    Format:
        [1] <resolution_summary>
            Thread: <thread_id> | Score: <score>
            → <thread_url>
    """
    lines: list[str] = []

    # Primary line: numbered resolution
    lines.append(f"**[{index}]** {citation.resolution_summary}")

    # Detail line: thread ID + optional score + optional date
    details: list[str] = []
    if citation.source_thread_id and citation.source_thread_id != "unknown":
        details.append(f"Thread: `{citation.source_thread_id}`")
    if citation.similarity_score is not None:
        details.append(f"Score: {citation.similarity_score:.2f}")
    if citation.resolved_at:
        details.append(f"Resolved: {citation.resolved_at.strftime('%Y-%m-%d')}")
    if details:
        lines.append(f"    {' | '.join(details)}")

    # URL line
    if citation.thread_url:
        lines.append(f"    → {citation.thread_url}")

    return "\n".join(lines)
