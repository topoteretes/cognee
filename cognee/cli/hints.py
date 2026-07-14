"""Next-step hints printed after the primary CLI commands succeed.

Each primary command (``remember``, ``recall``, ``cognify``, ``forget``)
ends with a single ``Next: ...`` line naming the natural follow-up command,
formatted so the user can copy-paste it. The copy lives here so it stays
consistent across commands and a later reword touches one file. Plain text,
no colour, so it composes cleanly with piped output.
"""

import cognee.cli.echo as fmt


def _next(command: str) -> None:
    """Print one copy-pastable next-step line."""
    fmt.echo(f"Next: {command}")


def hint_recall(dataset_name: str) -> None:
    """After ``remember`` or ``cognify``, the next step is to query the graph."""
    _next(f'cognee-cli recall "your question" -d {dataset_name}')


def hint_recall_empty(dataset_name: str) -> None:
    """After a ``recall`` that found nothing, point back at ``remember`` to seed the dataset."""
    _next(
        f'no matches in "{dataset_name}". Try "cognee-cli remember <path-or-text> '
        f'-d {dataset_name}" to seed it, then rerun this recall.'
    )


def hint_remember(dataset_name: str) -> None:
    """After ``forget``, the next step is to start fresh with ``remember``."""
    _next(f"cognee-cli remember <path-or-text> -d {dataset_name} to start a new session.")
