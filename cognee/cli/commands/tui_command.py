import argparse
from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


class TuiCommand(SupportsCliCommand):
    command_string = "tui"
    help_string = "Launch interactive Terminal User Interface"
    docs_url = DEFAULT_DOCS_URL

    description = """
Launch the Cognee Terminal User Interface (TUI).

The TUI provides an interactive, text-based interface for managing your
knowledge graphs with features like:

- **Context Management**: Add and manage data sources
- **Search & Query**: Interactive knowledge graph querying
- **Settings**: Configure API keys and models
- **Live Updates**: Real-time status and progress indicators

The TUI is keyboard-driven and supports:
- Arrow key navigation
- Keyboard shortcuts (h=Home, c=Context, s=Search, etc.)
- Tab completion for inputs

Perfect for managing Cognee from the terminal or SSH sessions!
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--no-mouse",
            action="store_true",
            help="Disable mouse support (keyboard only mode)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            fmt.echo("Starting Cognee TUI...")
            fmt.note("Press 'q' to quit, '?' for help")

            # Import and run TUI
            from cognee.cli.tui import run_tui

            run_tui(mouse=not args.no_mouse)

        except KeyboardInterrupt:
            fmt.note("\nTUI closed by user")
        except Exception as e:
            raise CliCommandException(f"Failed to start TUI: {str(e)}", error_code=1) from e
