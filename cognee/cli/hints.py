"""Next-step hints for the primary CLI commands.

Every primary command (``remember``, ``recall``, ``cognify``, ``forget``)
ends with a single line that names the next natural command the user
will typically want to run, formatted as copy-paste. Users driving the
CLI in a script or pipeline can suppress the hint with ``--quiet``.

The hints live in one module so the copy stays consistent across
commands and a follow-up docs pass can be made in one place instead of
grepping four command modules. Every hint is a plain string with no
ANSI colour so it composes cleanly with piped output.
"""

from __future__ import annotations

import cognee.cli.echo as fmt

_NEXT_LABEL = "Next"


def _hint(text: str) -> None:
    """Emit a single next-step line prefixed with ``Next:``.

    Kept tiny so callers just line up their argument with the copy in
    this module. ``fmt.echo`` is used so the sink honours the same
    stream configuration as the surrounding success block.
    """
    fmt.echo(f"{_NEXT_LABEL}: {text}")


def _quiet(args) -> bool:
    """Return True when the caller passed ``--quiet``.

    Silent-fail on missing attribute so a command that has not yet
    declared the flag on its parser does not raise; the hint just
    prints. All four primary commands declare it via
    ``add_quiet_flag(parser)``.
    """
    return bool(getattr(args, "quiet", False))


def add_quiet_flag(parser) -> None:
    """Attach the ``--quiet`` flag to a command's argparse parser.

    Kept as a helper so the option string and help text stay consistent
    across the four primary commands and a follow-up rename touches
    one place.
    """
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the next-step hint line (useful when piping or scripting).",
    )


def remember_hint(args, dataset_name: str) -> None:
    """Print the "recall next" hint after a successful ``remember``.

    Uses the user's actual dataset name so the copy is directly
    executable, not a placeholder.
    """
    if _quiet(args):
        return
    _hint(f'cognee-cli recall "your question" -d {dataset_name}')


def cognify_hint(args, dataset_name: str) -> None:
    """Print the "recall next" hint after a successful ``cognify``.

    Same target as :func:`remember_hint` because from the user's
    perspective both commands leave the same next step waiting.
    """
    if _quiet(args):
        return
    _hint(f'cognee-cli recall "your question" -d {dataset_name}')


def recall_hint(args, dataset_name: str, had_results: bool) -> None:
    """Print a "remember more" hint after ``recall``.

    Only fires when the recall returned no results, so a user who got a
    useful answer is not nudged toward a follow-up they did not ask
    for. When results were found the command already gives the user
    plenty to look at.
    """
    if _quiet(args):
        return
    if had_results:
        return
    _hint(
        f'no matches in "{dataset_name}". Try "cognee-cli remember <path-or-text> '
        f'-d {dataset_name}" to seed it, then rerun this recall.'
    )


def forget_hint(args, dataset_name: str) -> None:
    """Print the "remember to start fresh" hint after a successful ``forget``."""
    if _quiet(args):
        return
    _hint(f"cognee-cli remember <path-or-text> -d {dataset_name} to start a new session.")
