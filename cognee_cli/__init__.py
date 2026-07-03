"""Fast entry point for cognee-cli.

This package lives OUTSIDE the ``cognee`` package on purpose: importing any
module inside ``cognee`` runs ``cognee/__init__.py`` first, which configures
logging and imports the full API surface (~1.5-2s and several log lines before
argparse even sees ``--help``). The console script targets this shim instead,
so that:

- ``cognee-cli`` (no args), ``--help``/``-h``/``help`` and ``--version`` render
  in milliseconds without importing ``cognee`` at all, and
- every real command gets ``COGNEE_CLI_MODE=true`` exported *before* the
  package import, which routes logs to the log file and keeps the console
  reserved for the command's own output.

The static command table below is verified against the real parser by
``cognee/tests/cli_tests/test_fastpath_sync.py`` so it cannot silently drift
(the fate of the previous ``minimal_cli.py`` attempt).
"""

import os
import sys

DOCS_URL = "https://docs.cognee.ai"

# (command, one-line description) — grouped for the static help screen.
# Must stay in sync with the commands registered in cognee/cli/_cognee.py;
# a unit test enforces this.
COMMAND_GROUPS = [
    (
        "GETTING STARTED",
        [
            ("add", "Add files, text, or URLs to your memory"),
            ("cognify", "Build the knowledge graph from added data"),
            ("search", "Ask questions, get graph-grounded answers"),
            ("doctor", "Check that your setup is ready"),
        ],
    ),
    (
        "MEMORY",
        [
            ("remember", "Ingest data and build the knowledge graph in one call"),
            ("recall", "Search the knowledge graph for relevant information"),
            ("improve", "Enrich the graph with additional context and rules"),
            ("forget", "Remove data from the knowledge graph"),
        ],
    ),
    (
        "DATA & PIPELINE",
        [
            ("datasets", "Manage datasets (list, create, inspect, status, delete)"),
            ("delete", "Delete data from the knowledge base"),
            ("memify", "Run the memory enrichment pipeline on a dataset"),
            ("sessions", "View conversation sessions and Q&A history"),
            ("feedback", "Add or remove feedback on session Q&A entries"),
        ],
    ),
    (
        "CLOUD & SERVER",
        [
            ("serve", "Connect to a Cognee instance (cloud or local)"),
            ("push", "Upload a local dataset's knowledge graph to Cognee Cloud"),
        ],
    ),
    (
        "MAINTENANCE",
        [
            ("config", "Manage cognee configuration settings"),
            ("agents", "Manage agents (create, list, register, connections)"),
            ("upgrade", "Apply pending database migrations"),
            ("downgrade", "Revert data migrations to a revision"),
            ("history", "List the data-migration chain"),
            ("current", "Show each database's migration revision"),
            ("stamp", "Set the stored migration revision (bookkeeping repair)"),
        ],
    ),
]


def _use_color(stream) -> bool:
    """Color only when writing to a real terminal that wants it (no-color.org)."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


class _Style:
    def __init__(self, enabled: bool) -> None:
        self._on = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self._on else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("cognee")
    except Exception:
        return "unknown"


def _print_welcome() -> None:
    s = _Style(_use_color(sys.stdout))

    def row(command: str, description: str) -> str:
        # Pad on the plain text, then colorize — ANSI codes have no width.
        return f"  {s.cyan(command)}{' ' * max(2, 38 - len(command))}{description}"

    lines = [
        f"{s.bold('cognee ' + _version())} — memory for your AI agents",
        "",
        s.dim("Usage"),
        "  cognee-cli <command> [options]",
        "",
        s.dim("Get started"),
        row("cognee-cli add ./docs", "ingest files, text, or URLs"),
        row("cognee-cli cognify", "build your knowledge graph"),
        row('cognee-cli search "ask anything"', "graph-grounded answers"),
        "",
        f"{s.dim('Check your setup:')}  {s.cyan('cognee-cli doctor')}",
        f"{s.dim('All commands:')}      cognee-cli --help {s.dim('·')} {DOCS_URL}",
    ]
    sys.stdout.write("\n".join(lines) + "\n")


def _print_help() -> None:
    s = _Style(_use_color(sys.stdout))
    out = [
        "Give your AI agents durable memory from the command line.",
        "",
        s.bold("USAGE"),
        "  cognee-cli <command> [options]",
        "",
    ]
    for group, commands in COMMAND_GROUPS:
        out.append(s.bold(group))
        for name, desc in commands:
            out.append(f"  {name:<12} {desc}")
        out.append("")
    out += [
        s.bold("OPTIONS"),
        "  --version          Show the cognee version",
        "  --debug            Show full stack traces when a command fails",
        "  --verbose          Show detailed logs on the console",
        "  -ui                Start the cognee web UI",
        "  --user-id ID       Act as a specific user/agent (multi-agent isolation)",
        "  --api-url URL      Delegate commands to a running Cognee API server",
        "",
        s.bold("EXAMPLES"),
        "  $ cognee-cli add ./docs",
        "  $ cognee-cli cognify",
        '  $ cognee-cli search "What is this project about?"',
        "",
        s.bold("LEARN MORE"),
        "  cognee-cli <command> --help   detailed help for a command",
        f"  {DOCS_URL}",
    ]
    sys.stdout.write("\n".join(out) + "\n")


def _should_set_cli_mode(args) -> bool:
    """CLI mode routes logs to the log file and keeps the console for command
    output — right for one-shot commands, wrong for `-ui`, which launches
    long-running servers (backend, MCP, frontend): it would silence the
    launcher's own startup feedback and leak into the spawned backend via
    environment inheritance."""
    return "-ui" not in args


def main() -> int:
    args = sys.argv[1:]

    if not args:
        _print_welcome()
        return 0
    if args == ["--help"] or args == ["-h"] or args == ["help"]:
        _print_help()
        return 0
    if args == ["--version"] or args == ["-V"]:
        sys.stdout.write(f"cognee {_version()}\n")
        return 0

    # Real command: route logs to file, keep the console for command output.
    # Must be set before the first `import cognee` anywhere in the process.
    if _should_set_cli_mode(args):
        os.environ.setdefault("COGNEE_CLI_MODE", "true")

    try:
        from cognee.cli._cognee import main as full_main

        return full_main()
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        return 130
    except BrokenPipeError:
        # Downstream pipe closed (e.g. `cognee-cli ... | head`); exit quietly.
        try:
            sys.stderr.close()
        except Exception:
            pass
        return 141


def _main() -> None:
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
