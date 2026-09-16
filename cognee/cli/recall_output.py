"""Shared rendering for `cognee-cli recall`.

The in-process command and the ``--api-url`` dispatch print the same results,
so the label logic and the printer live here rather than in either lane.
"""

import cognee.cli.echo as fmt
from cognee.cli.code_search import print_code_results
from cognee.cli.config import COMPLETION_SEARCH_TYPES


def resolved_search_type(results, fallback: str) -> str:
    """Read the search type the SDK actually ran from the first result."""
    if not results:
        return fallback
    first = results[0]
    resolved = (
        first.get("search_type") if isinstance(first, dict) else getattr(first, "search_type", None)
    )
    if resolved is None:
        return fallback
    return getattr(resolved, "value", resolved)


def print_recall_results(results, fallback_type: str) -> None:
    """Pretty-print a non-empty recall result list.

    Each lane keeps its own empty-result handling; only the rendering is shared.
    """
    # Detect session results by _source tag
    if isinstance(results[0], dict) and results[0].get("_source") == "session":
        fmt.echo(f"\nFound {len(results)} session entry(ies):")
        fmt.echo("=" * 60)
        for i, entry in enumerate(results, 1):
            q = entry.get("question", "")
            a = entry.get("answer", "")
            t = entry.get("time", "")
            header = f"[{t}] " if t else ""
            if q:
                fmt.echo(f"{fmt.bold(f'{header}Q:')} {q}")
            if a:
                fmt.echo(f"{fmt.bold('A:')} {a}")
            if i < len(results):
                fmt.echo("-" * 40)
        return

    # Every branch keys off the type that actually ran, not the one requested:
    # an auto-routed call passes no type at all, and the SDK picks CHUNKS on its
    # own whenever no LLM is configured.
    resolved_type = resolved_search_type(results, fallback_type)
    fmt.echo(f"\nFound {len(results)} result(s) using {resolved_type}:")
    fmt.echo("=" * 60)

    if resolved_type in COMPLETION_SEARCH_TYPES:
        for i, result in enumerate(results, 1):
            fmt.echo(f"{fmt.bold('Response:')} {result}")
            if i < len(results):
                fmt.echo("-" * 40)
    elif resolved_type == "CHUNKS":
        for i, result in enumerate(results, 1):
            fmt.echo(f"{fmt.bold(f'Chunk {i}:')} {result}")
            fmt.echo()
    elif resolved_type == "CODE" and print_code_results(results):
        pass
    else:
        for i, result in enumerate(results, 1):
            fmt.echo(f"{fmt.bold(f'Result {i}:')} {result}")
            fmt.echo()
