"""Shared rendering for `cognee-cli recall`.

The in-process command and the ``--api-url`` dispatch print the same results,
so the label logic and the printer live here rather than in either lane.
"""

import cognee.cli.echo as fmt
from cognee.cli.code_search import print_code_results
from cognee.cli.config import COMPLETION_SEARCH_TYPES


def _field(entry, name: str):
    """Read one field off a recall entry.

    The in-process command gets RecallResponse models; the ``--api-url`` lane
    gets the dicts they were encoded to.
    """
    return entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)


def resolved_search_type(results, fallback: str) -> str:
    """Read the search type the SDK actually ran from the first result."""
    if not results:
        return fallback
    resolved = _field(results[0], "search_type")
    if resolved is None:
        return fallback
    return getattr(resolved, "value", resolved)


def _print_session_entries(entries) -> None:
    """Render session-cache Q&A entries."""
    fmt.echo(f"\nFound {len(entries)} session entry(ies):")
    fmt.echo("=" * 60)
    for i, entry in enumerate(entries, 1):
        q = _field(entry, "question") or ""
        a = _field(entry, "answer") or ""
        t = _field(entry, "time") or ""
        header = f"[{t}] " if t else ""
        if q:
            fmt.echo(f"{fmt.bold(f'{header}Q:')} {q}")
        if a:
            fmt.echo(f"{fmt.bold('A:')} {a}")
        if i < len(entries):
            fmt.echo("-" * 40)


def _print_graph_results(results, fallback_type: str) -> None:
    """Render everything that is not a session entry, keyed by search type."""
    # Keys off the type that actually ran, not the one requested: an auto-routed
    # call passes no type at all, and the SDK picks CHUNKS on its own whenever
    # no LLM is configured.
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


def print_recall_results(results, fallback_type: str) -> None:
    """Pretty-print a non-empty recall result list.

    Each lane keeps its own empty-result handling; only the rendering is shared.

    The list can mix sources: `recall()` with a session_id *and* datasets and no
    pinned type lets session and graph both contribute. Partition rather than
    branch on the first entry -- the tag is "source" in both lanes, and this
    branch used to read "_source", so it fired in neither. Fixing the name made
    it fire, and a single leading session entry would then have rendered the
    graph results as blank Q&A rows, dropping the answer the caller asked for.
    """
    session_entries = [entry for entry in results if _field(entry, "source") == "session"]
    other_results = [entry for entry in results if _field(entry, "source") != "session"]

    if session_entries:
        _print_session_entries(session_entries)
    if other_results:
        _print_graph_results(other_results, fallback_type)
