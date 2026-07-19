"""First-run guidance shared by local and API-backed CLI commands."""

from __future__ import annotations

import re
import shlex
from typing import Iterable

import cognee.cli.echo as fmt


def quote_cli_value(value: object) -> str:
    return shlex.quote(str(value))


def format_dataset_options(datasets: Iterable[str] | None) -> str:
    if not datasets:
        return ""

    quoted = " ".join(quote_cli_value(dataset) for dataset in datasets)
    return f" --datasets {quoted}"


def recall_command(dataset_name: str | None = None) -> str:
    dataset_options = format_dataset_options([dataset_name] if dataset_name else None)
    return f'cognee-cli recall "What should I remember?"{dataset_options}'


def remember_command() -> str:
    return 'cognee-cli remember "Cognee turns documents into AI memory."'


def echo_after_add(dataset_name: str) -> None:
    dataset = quote_cli_value(dataset_name)
    fmt.note(
        "Next: run "
        f"`cognee-cli cognify --datasets {dataset}`, then "
        f"`{recall_command(dataset_name)}`."
    )


def echo_after_cognify(datasets: Iterable[str] | None, background: bool = False) -> None:
    dataset_names = list(datasets or [])
    dataset_options = format_dataset_options(dataset_names)
    prefix = "When background processing finishes, run" if background else "Next: run"
    fmt.note(f'{prefix} `cognee-cli search "What is in this data?"{dataset_options}`.')


def echo_after_remember(
    dataset_name: str,
    dataset_id: str | None = None,
    background: bool = False,
) -> None:
    if background:
        if dataset_id:
            fmt.note(
                "Next: track progress with "
                f"`cognee-cli datasets status {quote_cli_value(dataset_id)}`. "
                f"When it finishes, run `{recall_command(dataset_name)}`."
            )
            return
        fmt.note(f"Next: when processing finishes, run `{recall_command(dataset_name)}`.")
        return

    fmt.note(f"Next: run `{recall_command(dataset_name)}`.")


def echo_after_forget() -> None:
    fmt.note(
        f"Next: add new memory with `{remember_command()}` or retry recall to confirm removal."
    )


def echo_empty_query_hint(command_name: str) -> None:
    query = "What is in this data?" if command_name == "search" else "What should I remember?"
    fmt.note(
        "Next: if this is your first run, create memory with "
        f'`{remember_command()}`, then retry `cognee-cli {command_name} "{query}"`.'
    )


_KEY_ERROR_PATTERN = re.compile(
    r"(llm_api_key|api[_ -]?key|authentication|unauthorized|invalid key)",
    re.IGNORECASE,
)


def with_first_run_error_hint(message: str) -> str:
    if not _KEY_ERROR_PATTERN.search(message):
        return message

    hint = (
        "Set LLM_API_KEY before running LLM-backed commands: "
        '`export LLM_API_KEY="..."`. '
        "If you only want to stage data first, run `cognee-cli add ...` before `cognee-cli cognify`."
    )

    if hint in message:
        return message

    return f"{message}\n{hint}"
