import argparse
import json
from typing import Optional, Tuple

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, Button
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding
from textual.coordinate import Coordinate


class EditModal(Screen):
    """Modal screen for editing a config value."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    EditModal {
        align: center middle;
    }

    #edit-dialog {
        width: 60;
        height: 13;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #edit-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #edit-key {
        color: $text-muted;
        margin-bottom: 1;
    }

    #edit-input {
        margin-bottom: 1;
    }

    #edit-buttons {
        align: center middle;
        height: 3;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, key: str, default_value: str):
        super().__init__()
        self.key = key
        self.default_value = default_value
        self.result = None

    def compose(self) -> ComposeResult:
        with Container(id="edit-dialog"):
            yield Label("Edit Configuration", id="edit-title")
            yield Label(f"Key: {self.key}", id="edit-key")
            yield Label(f"Default: {self.default_value}", id="edit-key")
            yield Input(placeholder="Enter new value", id="edit-input")
            with Horizontal(id="edit-buttons"):
                yield Button("Save", variant="primary", id="save-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            input_widget = self.query_one(Input)
            self.result = input_widget.value
            self.dismiss(self.result)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmModal(Screen):
    """Modal screen for confirming reset action."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ConfirmModal {
        align: center middle;
    }

    #confirm-dialog {
        width: 50;
        height: 11;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }

    #confirm-title {
        text-align: center;
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #confirm-message {
        text-align: center;
        margin-bottom: 2;
    }

    #confirm-buttons {
        align: center middle;
        height: 3;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, key: str, default_value: str):
        super().__init__()
        self.key = key
        self.default_value = default_value

    def compose(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label("⚠ Reset Configuration", id="confirm-title")
            yield Label(f"Reset '{self.key}' to default?", id="confirm-message")
            yield Label(f"Default value: {self.default_value}", id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Reset", variant="error", id="confirm-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ConfigTUIScreen(Screen):
    """Main config TUI screen."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "go_back", "Back"),
        Binding("e", "edit", "Edit"),
        Binding("r", "reset", "Reset"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
    ]

    CSS = """
    ConfigTUIScreen {
        background: $surface;
    }

    #config-header {
        dock: top;
        background: $boost;
        color: $text;
        content-align: center middle;
        text-style: bold;
        padding: 1;
        border: solid $primary;
    }

    #config-container {
        height: 100%;
        padding: 1;
    }

    DataTable {
        height: 1fr;
    }

    #config-footer {
        dock: bottom;
        height: 3;
        background: $boost;
        content-align: center middle;
        border: solid $primary;
    }
    """

    # Config key mappings with defaults (from existing config.py)
    CONFIG_KEYS = {
        "llm_provider": ("set_llm_provider", "openai"),
        "llm_model": ("set_llm_model", "gpt-5-mini"),
        "llm_api_key": ("set_llm_api_key", ""),
        "llm_endpoint": ("set_llm_endpoint", ""),
        "graph_database_provider": ("set_graph_database_provider", "kuzu"),
        "vector_db_provider": ("set_vector_db_provider", "lancedb"),
        "vector_db_url": ("set_vector_db_url", ""),
        "vector_db_key": ("set_vector_db_key", ""),
        "chunk_size": ("set_chunk_size", "1500"),
        "chunk_overlap": ("set_chunk_overlap", "10"),
    }

    def compose(self) -> ComposeResult:
        yield Label("🧠 cognee Config Manager", id="config-header")

        with Container(id="config-container"):
            table = DataTable()
            table.cursor_type = "row"
            table.zebra_stripes = True
            yield table

        yield Label(
            "[↑↓] Navigate  [e] Edit  [r] Reset  [Esc] Back  [q] Quit",
            id="config-footer"
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("KEY", "DEFAULT VALUE")

        # Add all config keys
        for key, (method, default) in self.CONFIG_KEYS.items():
            display_default = "(empty)" if default == "" else str(default)
            table.add_row(key, display_default)

        table.focus()

    def action_cursor_up(self) -> None:
        """Move cursor up in the table."""
        table = self.query_one(DataTable)
        table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in the table."""
        table = self.query_one(DataTable)
        table.action_cursor_down()

    def action_go_back(self) -> None:
        """Go back to main menu."""
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        """Quit the entire application."""
        self.app.exit()

    def action_edit(self) -> None:
        """Edit the selected config value."""
        table = self.query_one(DataTable)

        if table.cursor_coordinate.row < 0:
            return

        row_key = table.get_row_at(table.cursor_coordinate.row)
        key = str(row_key[0])
        default_value = str(row_key[1])

        def handle_edit_result(value: Optional[str]) -> None:
            if value is not None and value.strip():
                self._save_config(key, value)

        self.app.push_screen(EditModal(key, default_value), handle_edit_result)

    def action_reset(self) -> None:
        """Reset the selected config to default."""
        table = self.query_one(DataTable)

        if table.cursor_coordinate.row < 0:
            return

        row_key = table.get_row_at(table.cursor_coordinate.row)
        key = str(row_key[0])

        if key not in self.CONFIG_KEYS:
            return

        method_name, default_value = self.CONFIG_KEYS[key]
        display_default = "(empty)" if default_value == "" else str(default_value)

        def handle_confirm_result(confirmed: bool) -> None:
            if confirmed:
                self._reset_config(key, method_name, default_value)

        self.app.push_screen(
            ConfirmModal(key, display_default),
            handle_confirm_result
        )

    def _save_config(self, key: str, value: str) -> None:
        """Save config value using cognee.config.set()."""
        try:
            import cognee

            # Try to parse as JSON (numbers, booleans, etc)
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                parsed_value = value

            cognee.config.set(key, parsed_value)
            self.notify(f"✓ Set {key} = {parsed_value}", severity="information")

        except Exception as e:
            self.notify(f"✗ Failed to set {key}: {str(e)}", severity="error")

    def _reset_config(self, key: str, method_name: str, default_value: any) -> None:
        """Reset config to default using the mapped method."""
        try:
            import cognee

            method = getattr(cognee.config, method_name)
            method(default_value)

            display_default = "(empty)" if default_value == "" else str(default_value)
            self.notify(
                f"✓ Reset {key} to default: {display_default}",
                severity="information"
            )

        except Exception as e:
            self.notify(f"✗ Failed to reset {key}: {str(e)}", severity="error")


class ConfigTUICommand(SupportsCliCommand):
    """TUI command for config management."""

    command_string = "config-tui"
    help_string = "Launch interactive TUI for managing cognee configuration"
    docs_url = f"{DEFAULT_DOCS_URL}/usage/config-tui"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        try:
            class ConfigTUIApp(App):
                """Simple app wrapper for config TUI."""

                def on_mount(self) -> None:
                    self.push_screen(ConfigTUIScreen())

            app = ConfigTUIApp()
            app.run()

        except ImportError:
            raise CliCommandException(
                "Textual is not installed. Install with: pip install textual",
                docs_url=self.docs_url,
            )
        except Exception as ex:
            raise CliCommandException(
                f"Failed to launch config TUI: {str(ex)}",
                docs_url=self.docs_url,
                raiseable_exception=ex,
            )