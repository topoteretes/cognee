import argparse
from cognee.cli import SupportsCliCommand
from cognee.cli.config import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException
from cognee.cli.tui.home_screen import HomeScreen


class TuiCommand(SupportsCliCommand):
    @property
    def command_string(self) -> str:
        return "tui"

    @property
    def help_string(self) -> str:
        return "Launch the interactive Textual TUI for cognee commands"

    @property
    def docs_url(self) -> str:
        return f"{DEFAULT_DOCS_URL}/usage/tui"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        # No additional arguments for now
        pass

    def execute(self, args: argparse.Namespace) -> None:
        try:
            from textual.app import App
            from cognee.shared.logging_utils import setup_logging
            class CogneeTUI(App):
                """Main TUI application for cognee."""

                CSS = """
                Screen {
                    background: $surface;
                }
                """

                def on_mount(self) -> None:
                    """Push the home screen on mount."""
                    self.push_screen(HomeScreen())

            setup_logging(enable_console_logging=False)
            app = CogneeTUI()
            app.run()
        except ImportError:
            raise CliCommandException(
                "Textual is not installed. Install with: pip install textual",
                docs_url=self.docs_url,
            )
        except Exception as ex:
            raise CliCommandException(
                f"Failed to launch TUI: {str(ex)}",
                docs_url=self.docs_url,
                raiseable_exception=ex,
            )
