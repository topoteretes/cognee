"""CLI output formatting utilities"""

import sys
import click
from typing import Any


def echo(message: str = "", color: str = None, err: bool = False) -> None:
    """Echo a message to stdout or stderr with optional color"""
    click.secho(message, fg=color, err=err)


def note(message: str) -> None:
    """Print a note in blue"""
    echo(f"Note: {message}", color="blue")


def warning(message: str) -> None:
    """Print a warning in yellow"""
    echo(f"Warning: {message}", color="yellow")


def error(message: str) -> None:
    """Print an error in red"""
    echo(f"Error: {message}", color="red", err=True)


def success(message: str) -> None:
    """Print a success message in green"""
    echo(f"Success: {message}", color="green")


def bold(text: str) -> str:
    """Make text bold"""
    return click.style(text, bold=True)


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation"""
    return click.confirm(message, default=default)


def prompt(message: str, default: Any = None) -> str:
    """Prompt user for input"""
    return click.prompt(message, default=default)
