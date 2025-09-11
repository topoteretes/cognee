import sys
import os
import argparse
import signal
import subprocess
from typing import Any, Sequence, Dict, Type, cast, List
import click

try:
    import rich_argparse
    from rich.markdown import Markdown

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from cognee.cli import SupportsCliCommand, DEFAULT_DOCS_URL
from cognee.cli.config import CLI_DESCRIPTION
from cognee.cli import debug
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


ACTION_EXECUTED = False


def print_help(parser: argparse.ArgumentParser) -> None:
    if not ACTION_EXECUTED:
        parser.print_help()


class DebugAction(argparse.Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: Any = argparse.SUPPRESS,
        default: Any = argparse.SUPPRESS,
        help: str = None,
    ) -> None:
        super(DebugAction, self).__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0, help=help
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str = None,
    ) -> None:
        # Enable debug mode for stack traces
        debug.enable_debug()
        fmt.note("Debug mode enabled. Full stack traces will be shown.")


class UiAction(argparse.Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: Any = argparse.SUPPRESS,
        default: Any = argparse.SUPPRESS,
        help: str = None,
    ) -> None:
        super(UiAction, self).__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0, help=help
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str = None,
    ) -> None:
        # Set a flag to indicate UI should be started
        global ACTION_EXECUTED
        ACTION_EXECUTED = True
        namespace.start_ui = True


# Debug functionality is now in cognee.cli.debug module


def _discover_commands() -> List[Type[SupportsCliCommand]]:
    """Discover all available CLI commands"""
    # Import commands dynamically to avoid early cognee initialization
    commands = []

    command_modules = [
        ("cognee.cli.commands.add_command", "AddCommand"),
        ("cognee.cli.commands.search_command", "SearchCommand"),
        ("cognee.cli.commands.cognify_command", "CognifyCommand"),
        ("cognee.cli.commands.delete_command", "DeleteCommand"),
        ("cognee.cli.commands.config_command", "ConfigCommand"),
    ]

    for module_path, class_name in command_modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            command_class = getattr(module, class_name)
            commands.append(command_class)
        except (ImportError, AttributeError) as e:
            fmt.warning(f"Failed to load command {class_name}: {e}")

    return commands


def _create_parser() -> tuple[argparse.ArgumentParser, Dict[str, SupportsCliCommand]]:
    parser = argparse.ArgumentParser(
        description=f"{CLI_DESCRIPTION} Further help is available at {DEFAULT_DOCS_URL}."
    )

    # Get version dynamically
    try:
        from cognee.version import get_cognee_version

        version = get_cognee_version()
    except ImportError:
        version = "unknown"

    parser.add_argument("--version", action="version", version=f"cognee {version}")
    parser.add_argument(
        "--debug",
        action=DebugAction,
        help="Enable debug mode to show full stack traces on exceptions",
    )
    parser.add_argument(
        "-ui",
        action=UiAction,
        help="Start the cognee web UI interface",
    )

    subparsers = parser.add_subparsers(title="Available commands", dest="command")

    # Discover and install commands
    command_classes = _discover_commands()
    installed_commands: Dict[str, SupportsCliCommand] = {}

    for command_class in command_classes:
        command = command_class()
        if command.command_string in installed_commands:
            continue

        command_parser = subparsers.add_parser(
            command.command_string,
            help=command.help_string,
            description=command.description if hasattr(command, "description") else None,
        )
        command.configure_parser(command_parser)
        installed_commands[command.command_string] = command

    # Add rich formatting if available
    if HAS_RICH:

        def add_formatter_class(parser: argparse.ArgumentParser) -> None:
            parser.formatter_class = rich_argparse.RichHelpFormatter

            if parser.description:
                parser.description = Markdown(parser.description, style="argparse.text")
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for _subcmd, subparser in action.choices.items():
                        add_formatter_class(subparser)

        add_formatter_class(parser)

    return parser, installed_commands


def main() -> int:
    """Main CLI entry point"""
    parser, installed_commands = _create_parser()
    args = parser.parse_args()

    # Handle UI flag
    if hasattr(args, "start_ui") and args.start_ui:
        server_process = None

        def signal_handler(signum, frame):
            """Handle Ctrl+C and other termination signals"""
            nonlocal server_process
            fmt.echo("\nShutting down UI server...")
            if server_process:
                try:
                    # Try graceful termination first
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=5)
                        fmt.success("UI server stopped gracefully.")
                    except subprocess.TimeoutExpired:
                        # If graceful termination fails, force kill
                        fmt.echo("Force stopping UI server...")
                        server_process.kill()
                        server_process.wait()
                        fmt.success("UI server stopped.")
                except Exception as e:
                    fmt.warning(f"Error stopping server: {e}")
            sys.exit(0)

        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request

        try:
            from cognee import start_ui

            fmt.echo("Starting cognee UI...")
            server_process = start_ui(host="localhost", port=3001, open_browser=True)

            if server_process:
                fmt.success("UI server started successfully!")
                fmt.echo("The interface is available at: http://localhost:3001")
                fmt.note("Press Ctrl+C to stop the server...")

                try:
                    # Keep the server running
                    import time

                    while server_process.poll() is None:  # While process is still running
                        time.sleep(1)
                except KeyboardInterrupt:
                    # This shouldn't happen now due to signal handler, but kept for safety
                    signal_handler(signal.SIGINT, None)

                return 0
            else:
                fmt.error("Failed to start UI server. Check the logs above for details.")
                return 1

        except Exception as ex:
            fmt.error(f"Error starting UI: {str(ex)}")
            if debug.is_debug_enabled():
                raise ex
            return 1

    if cmd := installed_commands.get(args.command):
        try:
            cmd.execute(args)
        except Exception as ex:
            docs_url = cmd.docs_url if hasattr(cmd, "docs_url") else DEFAULT_DOCS_URL
            error_code = -1
            raiseable_exception = ex

            # Handle CLI-specific exceptions
            if isinstance(ex, CliCommandException):
                error_code = ex.error_code
                docs_url = ex.docs_url or docs_url
                raiseable_exception = ex.raiseable_exception

            # Print exception
            if raiseable_exception:
                fmt.error(str(ex))

            fmt.note(f"Please refer to our docs at '{docs_url}' for further assistance.")

            if debug.is_debug_enabled() and raiseable_exception:
                raise raiseable_exception

            return error_code
    else:
        print_help(parser)
        return -1

    return 0


def _main() -> None:
    """Script entry point"""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
