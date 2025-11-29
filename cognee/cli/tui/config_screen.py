import argparse
import json

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.exceptions import CliCommandException

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Input, Label, Button, Static
from textual.containers import Container, Horizontal
from textual.binding import Binding

from cognee.cli.tui.base_screen import BaseTUIScreen


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


class ConfigTUIScreen(BaseTUIScreen):
    """Main config TUI screen with inline editing."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "cancel_or_back", "Back/Cancel"),
        Binding("e", "edit", "Edit"),
        Binding("enter", "confirm_edit", "Confirm", show=False),
        Binding("r", "reset", "Reset"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
    ]

    CSS = BaseTUIScreen.CSS + """
    ConfigTUIScreen {
        background: $surface;
    }

    #config-container {
        height: 100%;
        padding: 1;
    }

    DataTable {
        height: 1fr;
    }

    #inline-edit-container {
        display: none;
        height: auto;
        padding: 0 1;
        margin-top: 1;
    }

    #inline-edit-container.visible {
        display: block;
    }

    #edit-label {
        color: $text-muted;
        margin-bottom: 0;
    }

    #inline-input {
        width: 100%;
    }

    #config-footer {
        dock: bottom;
        padding: 1 0;
        background: $boost;
        color: $text-muted;
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

    def __init__(self):
        super().__init__()
        self.editing_key = None  # Track which key is being edited

    def compose_content(self) -> ComposeResult:
        with Container(id="config-container"):
            table = DataTable()
            table.cursor_type = "row"
            table.zebra_stripes = True
            yield table
            with Container(id="inline-edit-container"):
                yield Label("", id="edit-label")
                yield Input(placeholder="Enter new value", id="inline-input")

    def compose_footer(self) -> ComposeResult:
        yield Static(
            "↑↓: Navigate  •  e: Edit  •  Enter: Save  •  r: Reset  •  Esc: Back  •  q: Quit",
            id="config-footer"
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("KEY", "VALUE")

        # Add all config keys
        for key, (method, default) in self.CONFIG_KEYS.items():
            display_default = "(empty)" if default == "" else str(default)
            table.add_row(key, display_default)

        table.focus()

    def action_cursor_up(self) -> None:
        """Move cursor up in the table."""
        if self.editing_key:
            return  # Don't navigate while editing
        table = self.query_one(DataTable)
        table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in the table."""
        if self.editing_key:
            return  # Don't navigate while editing
        table = self.query_one(DataTable)
        table.action_cursor_down()

    def action_cancel_or_back(self) -> None:
        """Cancel editing or go back to main menu."""
        if self.editing_key:
            self._cancel_edit()
        else:
            self.app.pop_screen()

    def action_quit_app(self) -> None:
        """Quit the entire application."""
        self.app.exit()

    def action_edit(self) -> None:
        """Start inline editing for the selected config value."""
        if self.editing_key:
            return  # Already editing

        table = self.query_one(DataTable)
        if table.cursor_coordinate.row < 0:
            return

        row_data = table.get_row_at(table.cursor_coordinate.row)
        key = str(row_data[0])
        default_value = str(row_data[1])

        self.editing_key = key

        # Show the inline edit container
        edit_container = self.query_one("#inline-edit-container")
        edit_container.add_class("visible")

        # Update label and input
        label = self.query_one("#edit-label", Label)
        label.update(f"Editing: {key} (default: {default_value})")

        input_widget = self.query_one("#inline-input", Input)
        input_widget.value = ""
        input_widget.placeholder = f"Enter new value for {key}"
        input_widget.focus()

    def action_confirm_edit(self) -> None:
        """Confirm the inline edit and save the value."""
        if not self.editing_key:
            return

        input_widget = self.query_one("#inline-input", Input)
        value = input_widget.value.strip()

        if value:
            self._save_config(self.editing_key, value)

        self._cancel_edit()

    def _cancel_edit(self) -> None:
        """Cancel the current edit and hide the input."""
        self.editing_key = None

        # Hide the inline edit container
        edit_container = self.query_one("#inline-edit-container")
        edit_container.remove_class("visible")

        # Clear input
        input_widget = self.query_one("#inline-input", Input)
        input_widget.value = ""

        # Return focus to table
        table = self.query_one(DataTable)
        table.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input field."""
        if event.input.id == "inline-input" and self.editing_key:
            self.action_confirm_edit()

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
