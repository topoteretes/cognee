import argparse
from cognee.cli import SupportsCliCommand
from cognee.cli.config import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException
from cognee.version import get_cognee_version
from textual.app import App, ComposeResult
from textual.widgets import ListView, ListItem, Static
from textual.containers import Container, Horizontal
from textual.binding import Binding


def make_item(icon, command, description):
    # Compose a ListItem that contains a Horizontal container with 3 children
    return ListItem(
        Horizontal(
            Static(icon, classes="cmd-icon"),
            Static(command, classes="cmd-name"),
            Static(description, classes="cmd-desc"),
            classes="cmd-row",
        )
    )


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
                    border: solid $primary;
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
                    padding: 0 10;
                    border: solid $accent;
                    margin-bottom: 2;
                }
                
                ListView > ListItem {
                    width: 100%;
                    padding: 0;
                    margin: 0;
                }

                ListView {
                    height: auto;
                    background: $surface;
                    border: none;
                    padding: 0 0;
                }

                ListItem {
                    background: $surface;
                    color: $text;
                    padding: 0 1;
                    height: auto;
                    width: 100%;
                }
                
                ListItem.highlighted {
                    background: $primary-darken-2;
                }
                
                CommandItem {
                    width: 100%;
                    background: transparent;
                }

                #footer-info {
                    dock: bottom;
                    height: 3;
                    background: $boost;
                    color: $text-muted;
                    content-align: center middle;
                    border: solid $primary;
                }
                
                .cmd-row {
                    width: 100%;
                    height: auto;
                    align-horizontal: left;
                    padding: 0 1;
                }
                
                .cmd-icon {
                    width: 4;
                    text-align: center;
                }
                
                .cmd-name {
                    width: 14;
                    padding-left: 1;
                }
                
                .cmd-desc {
                    width: 1fr;
                    overflow: auto;
                    padding-left: 1;
                }
                
                """

                BINDINGS = [
                    Binding("q", "quit", "Quit", priority=True),
                    Binding("escape", "quit", "Quit", priority=True),
                    Binding("enter", "select", "Select", priority=True),
                    Binding("up", "nav_up", "Up", priority=True),
                    Binding("down", "nav_down", "Down", priority=True),
                ]

                def __init__(self):
                    super().__init__()
                    self.lv = None
                    self.current_index = 0

                def compose(self) -> ComposeResult:
                    version = get_cognee_version()
                    yield Static(f"🧠 cognee v{version}", id="header")

                    with Container(id="main-container"):
                        with Container(id="title-wrapper"):
                            yield Static("Select Command", id="title")
                        yield ListView(
                            make_item("📥", "add", "Add data to cognee"),
                            make_item("🔍", "search", "Search data in cognee"),
                            make_item("⚡", "cognify", "Process data in cognee"),
                            make_item("🗑️", "delete", "Delete data from cognee"),
                            make_item("⚙️", "config", "Configure cognee settings"),
                        )

                    yield Static(
                        "↑↓: Navigate  •  Enter: Select  •  q/Esc: Quit",
                        id="footer-info"
                    )

                def on_mount(self) -> None:
                    """Focus the list view on mount."""
                    self.lv = self.query_one(ListView)
                    self.current_index = 0
                    self.set_focus(self.lv)
                    self._apply_highlight()

                def _apply_highlight(self) -> None:
                    lv = self.lv
                    children = list(lv.children)
                    self.lv.index = self.current_index
                    for idx, item in enumerate(children):
                        if idx == self.current_index:
                            item.add_class("highlighted")
                        else:
                            item.remove_class("highlighted")

                def action_nav_up(self) -> None:
                    self.current_index = max(0, self.current_index - 1)
                    self._apply_highlight()

                def action_nav_down(self) -> None:
                    children = list(self.lv.children)
                    self.current_index = min(len(children) - 1, self.current_index + 1)
                    self._apply_highlight()

                def on_list_view_selected(self, event: ListView.Selected) -> None:
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