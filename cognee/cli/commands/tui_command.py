import argparse
from cognee.cli import SupportsCliCommand
from cognee.cli.config import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException
from cognee.version import get_cognee_version

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
            from textual.app import App, ComposeResult
            from textual.widgets import Header, Footer, ListView, ListItem, Static
            from textual.containers import Container, Vertical
            from textual.binding import Binding

            class CommandItem(Static):
                """A custom widget for command items with icon and description."""

                def __init__(self, icon: str, command: str, description: str):
                    self.icon = icon
                    self.command = command
                    self.description = description
                    super().__init__()

                def render(self) -> str:
                    return f"{self.icon}  {self.command:<12} {self.description}"

            class CogneeTUI(App):
                """A k9s-style TUI for cognee commands."""

                CSS = """
                Screen {
                    background: $surface;
                }

                #header {
                    dock: top;
                    height: 3;
                    background: $boost;
                    color: $text;
                    content-align: center middle;
                    border: solid $primary;
                }

                #main-container {
                    height: 100%;
                    border: thick $primary;
                    background: $surface;
                    padding: 1;
                }
                #title-wrapper {
                    width: 100%;
                    height: auto;
                    align: center middle; 
                }

                #title {
                    text-align: center;
                    width: auto;
                    color: $accent;
                    text-style: bold;
                    padding: 0 3;
                    border: solid $accent;
                    margin-bottom: 2;
                }

                ListView {
                    height: auto;
                    background: $surface;
                    border: none;
                    padding: 0 2;
                }

                ListItem {
                    background: $surface;
                    color: $text;
                    padding: 0 1;
                    height: auto;
                }

                ListItem:hover {
                    background: $surface;
                }

                ListItem.--highlight {
                    background: $primary;
                    color: $text;
                }

                CommandItem {
                    width: 100%;
                }

                #footer-info {
                    dock: bottom;
                    height: 3;
                    background: $boost;
                    color: $text-muted;
                    content-align: center middle;
                    border: solid $primary;
                }
                """

                BINDINGS = [
                    Binding("q", "quit", "Quit", priority=True),
                    Binding("escape", "quit", "Quit", priority=True),
                    Binding("enter", "select", "Select", priority=True),
                ]

                def compose(self) -> ComposeResult:
                    version = get_cognee_version()
                    yield Static(f"🧠 cognee v{version}", id="header")

                    with Container(id="main-container"):
                        with Container(id="title-wrapper"):
                            yield Static("Select Command", id="title")
                        yield ListView(
                            ListItem(CommandItem("📥", "add", "Add data to cognee")),
                            ListItem(CommandItem("🔍", "search", "Search data in cognee")),
                            ListItem(CommandItem("⚡", "cognify", "Process data in cognee")),
                            ListItem(CommandItem("🗑️", "delete", "Delete data from cognee")),
                            ListItem(CommandItem("⚙️", "config", "Configure cognee settings")),
                        )

                    yield Static(
                        "↑↓: Navigate  •  Enter: Select  •  q/Esc: Quit",
                        id="footer-info"
                    )

                def on_mount(self) -> None:
                    """Focus the list view on mount."""
                    self.query_one(ListView).index = 0

                def on_list_view_selected(self, event: ListView.Selected) -> None:
                    """Handle command selection."""
                    command_item = event.item.query_one(CommandItem)
                    command = command_item.command
                    fmt.echo(f"Selected command: {command}")
                    self.exit()

                def action_select(self) -> None:
                    """Select the current item."""
                    list_view = self.query_one(ListView)
                    list_view.action_select_cursor()

            app = CogneeTUI()
            app.run()
            fmt.success("TUI exited successfully!")
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