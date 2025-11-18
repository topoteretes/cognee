"""Home Screen for Cognee TUI"""

from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Button, Static
from textual.containers import Container, Vertical


class HomeScreen(Screen):
    """Main dashboard screen"""

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="menu-container", classes="center"):
            yield Static("[bold cyan]🧠 Cognee Context Manager[/bold cyan]", classes="title")
            yield Static("\nManage your AI memory and context\n", classes="center")

            with Vertical():
                yield Button("📁 Manage Context", id="context", variant="primary")
                yield Button("🔍 Search & Query", id="query", variant="success")
                yield Button("⚙️  Settings", id="settings", variant="default")
                yield Button("❓ Help", id="help", variant="default")
                yield Button("🚪 Exit", id="exit", variant="error")

            yield Static("\n[dim]Use arrow keys • Enter to select[/dim]", classes="center")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "context":
            from cognee.cli.tui.screens.context import ContextScreen

            self.app.push_screen(ContextScreen())

        elif button_id == "query":
            from cognee.cli.tui.screens.query import QueryScreen

            self.app.push_screen(QueryScreen())

        elif button_id == "settings":
            from cognee.cli.tui.screens.settings import SettingsScreen

            self.app.push_screen(SettingsScreen())

        elif button_id == "help":
            self.app.action_help()

        elif button_id == "exit":
            self.app.exit()

    def on_mount(self) -> None:
        # Ensure initial focus so arrow keys can move between buttons
        try:
            first_button = self.query_one("#context", Button)
            first_button.focus()
        except Exception:
            # If the button isn't found for any reason, ignore
            pass
