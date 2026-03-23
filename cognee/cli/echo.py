"""CLI output formatting utilities"""

import sys
import click
from typing import Any

from cognee.cli.exceptions import CliCommandException

_JSON_MODE = False


def enable_json_mode() -> None:
    """Enable JSON mode — suppresses human-readable stdout, redirects diagnostics to stderr"""
    global _JSON_MODE
    _JSON_MODE = True


def is_json_mode() -> bool:
    """Check if JSON mode is enabled"""
    return _JSON_MODE


def echo(message: str = "", color: str = None, err: bool = False) -> None:
    """Echo a message to stdout or stderr with optional color.
    In JSON mode, stdout output is suppressed (stderr still works)."""
    if _JSON_MODE and not err:
        return
    click.secho(message, fg=color, err=err)


def note(message: str) -> None:
    """Print a note in blue. In JSON mode, goes to stderr."""
    if _JSON_MODE:
        click.secho(f"Note: {message}", fg="blue", err=True)
    else:
        echo(f"Note: {message}", color="blue")


def warning(message: str) -> None:
    """Print a warning in yellow. In JSON mode, goes to stderr."""
    if _JSON_MODE:
        click.secho(f"Warning: {message}", fg="yellow", err=True)
    else:
        echo(f"Warning: {message}", color="yellow")


def error(message: str) -> None:
    """Print an error in red (always stderr)"""
    echo(f"Error: {message}", color="red", err=True)


def success(message: str) -> None:
    """Print a success message in green. In JSON mode, goes to stderr."""
    if _JSON_MODE:
        click.secho(f"Success: {message}", fg="green", err=True)
    else:
        echo(f"Success: {message}", color="green")


def bold(text: str) -> str:
    """Make text bold"""
    return click.style(text, bold=True)


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation. Raises in JSON mode."""
    if _JSON_MODE:
        raise CliCommandException(
            "Interactive confirmation required; use --force with --json",
            error_code=2,
        )
    return click.confirm(message, default=default)


def prompt(message: str, default: Any = None) -> str:
    """Prompt user for input. Raises in JSON mode."""
    if _JSON_MODE:
        raise CliCommandException(
            "Interactive prompt not supported in --json mode",
            error_code=2,
        )
    return click.prompt(message, default=default)
