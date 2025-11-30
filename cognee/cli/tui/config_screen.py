import argparse
import json
from typing import Any, Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.exceptions import CliCommandException

try:
    from textual.app import App, ComposeResult
    from textual.screen import Screen
    from textual.widgets import DataTable, Input, Label, Button, Static
    from textual.containers import Container, Horizontal
    from textual.binding import Binding
    from cognee.cli.tui.base_screen import BaseTUIScreen
except ImportError:
    # Handle case where textual is not installed to prevent import errors at module level
    BaseTUIScreen = object


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
        width: 60;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }

    #confirm-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #confirm-message {
        text-align: center;
        margin-bottom: 2;
    }

    .tui-dialog-buttons {
        align: center middle;
        height: auto;
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
            yield Label(f"Are you sure you want to reset '{self.key}'?", id="confirm-message")
            yield Label(f"It will revert to: {self.default_value}", classes="dim-text")

            with Horizontal(classes="tui-dialog-buttons"):
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
    """Main config TUI screen with inline editing and live data fetching."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "cancel_or_back", "Back/Cancel"),
        Binding("e", "edit", "Edit"),
        Binding("enter", "confirm_edit", "Confirm", show=False),
        Binding("r", "reset", "Reset"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
    ]

    CSS = (
        BaseTUIScreen.CSS
        + """
    DataTable {
        height: 1fr;
        text-align: center;
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
        margin-bottom: 1;
    }

    #inline-input {
        width: 100%;
    }

    .dim-text {
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }
    """
    )

    # Config key mappings: Key -> (Reset Method Name, Default Value)
    CONFIG_MAP = {
        "llm_provider": ("set_llm_provider", "openai"),
        "llm_model": ("set_llm_model", "gpt-5-mini"),
        "llm_api_key": ("set_llm_api_key", ""),
        "llm_endpoint": ("set_llm_endpoint", ""),
        "graph_database_provider": ("set_graph_database_provider", "kuzu"),
        "vector_db_provider": ("set_vector_db_provider", "lancedb"),
        "vector_db_url": ("set_vector_db_url", ""),
        "vector_db_key": ("set_vector_db_key", ""),
        "chunk_size": ("set_chunk_size", 1500),
        "chunk_overlap": ("set_chunk_overlap", 10),
    }

    def __init__(self):
        super().__init__()
        self.editing_key = None  # Track which key is being edited

    def compose_content(self) -> ComposeResult:
        with Container(classes="tui-main-container"):
            with Container(classes="tui-title-wrapper"):
                yield Static("⚙️  Configuration Manager", classes="tui-title-bordered")

            with Container(classes="tui-bordered-wrapper"):
                table = DataTable(id="config-table")
                table.cursor_type = "row"
                table.zebra_stripes = True
                yield table

                with Container(id="inline-edit-container"):
                    yield Label("", id="edit-label")
                    yield Input(placeholder="Enter new value", id="inline-input")

    def compose_footer(self) -> ComposeResult:
        yield Static(
            "↑↓: Navigate  •  e: Edit  •  Enter: Save  •  r: Reset  •  Esc: Back  •  q: Quit",
            classes="tui-footer",
        )

    def on_mount(self) -> None:
        """Initialize the table with columns and current data."""
        table = self.query_one(DataTable)
        table.add_columns("Configuration Key", "Current Value")

        self._load_table_data()
        table.focus()

    def _load_table_data(self) -> None:
        """Fetch real config values and populate the table."""
        table = self.query_one(DataTable)
        table.clear()

        try:
            import cognee

            # Check if get method exists, otherwise warn
            has_get = hasattr(cognee.config, "get")
        except ImportError:
            has_get = False
            self.notify("Could not import cognee config", severity="error")

        for key, (_, default_val) in self.CONFIG_MAP.items():
            value_display = "N/A"

            if has_get:
                try:
                    raw_val = cognee.config.get(key)
                    if raw_val is None:
                        raw_val = default_val
                    value_display = str(raw_val) if raw_val is not None else "(empty)"
                except Exception:
                    value_display = "Error fetching value"

            table.add_row(key, value_display, key=key)

    def action_cursor_up(self) -> None:
        if self.editing_key:
            return
        self.query_one(DataTable).action_cursor_up()

    def action_cursor_down(self) -> None:
        if self.editing_key:
            return
        self.query_one(DataTable).action_cursor_down()

    def action_cancel_or_back(self) -> None:
        if self.editing_key:
            self._cancel_edit()
        else:
            self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_edit(self) -> None:
        """Start inline editing for the selected config value."""
        if self.editing_key:
            return

        table = self.query_one(DataTable)
        if table.cursor_row < 0:
            return

        # Get row data using the cursor
        row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        current_val = table.get_cell(row_key, list(table.columns.keys())[1])  # Get value column

        self.editing_key = str(row_key.value)

        # Show edit container
        edit_container = self.query_one("#inline-edit-container")
        edit_container.add_class("visible")

        # Update UI
        label = self.query_one("#edit-label", Label)
        label.update(f"Editing: [bold]{self.editing_key}[/bold]")

        input_widget = self.query_one("#inline-input", Input)
        input_widget.value = ""
        # Don't put "empty" or "N/A" into the input box to save user deleting it
        if current_val not in ["(empty)", "N/A", "Error fetching value"]:
            input_widget.value = str(current_val)

        input_widget.placeholder = f"Enter new value for {self.editing_key}"
        input_widget.focus()

    def action_confirm_edit(self) -> None:
        """Confirm the inline edit and save the value."""
        if not self.editing_key:
            return

        input_widget = self.query_one("#inline-input", Input)
        value = input_widget.value.strip()

        # Allow saving even if empty (might mean unset/empty string)
        self._save_config(self.editing_key, value)
        self._cancel_edit()

    def _cancel_edit(self) -> None:
        self.editing_key = None
        edit_container = self.query_one("#inline-edit-container")
        edit_container.remove_class("visible")
        self.query_one("#inline-input", Input).value = ""
        self.query_one(DataTable).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "inline-input" and self.editing_key:
            self.action_confirm_edit()

    def action_reset(self) -> None:
        """Reset the selected config to default."""
        table = self.query_one(DataTable)
        if table.cursor_row < 0:
            return

        row_key_obj = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        key = str(row_key_obj.value)

        if key not in self.CONFIG_MAP:
            self.notify(f"Cannot reset '{key}'", severity="warning")
            return

        _, default_value = self.CONFIG_MAP[key]
        display_default = "(empty)" if default_value == "" else str(default_value)

        def handle_confirm_result(confirmed: bool) -> None:
            if confirmed:
                self._reset_config(key)

        self.app.push_screen(ConfirmModal(key, display_default), handle_confirm_result)

    def _save_config(self, key: str, value: str) -> None:
        """Save config value and update UI."""
        try:
            import cognee

            # Parse value types (restore JSON behavior)
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # If it looks like a boolean but json didn't catch it
                if value.lower() == "true":
                    parsed_value = True
                elif value.lower() == "false":
                    parsed_value = False
                else:
                    parsed_value = value

            cognee.config.set(key, parsed_value)
            self._update_table_row(key, parsed_value)
            self.notify(f"✓ Set {key}", severity="information")

        except Exception as e:
            self.notify(f"✗ Error setting {key}: {str(e)}", severity="error")

    def _reset_config(self, key: str) -> None:
        """Reset config to default using mapped method and update UI."""
        try:
            import cognee

            method_name, default_value = self.CONFIG_MAP[key]

            if hasattr(cognee.config, method_name):
                method = getattr(cognee.config, method_name)
                method(default_value)

                # IMPROVEMENT: Update table immediately
                self._update_table_row(key, default_value)
                self.notify(f"✓ Reset {key}", severity="information")
            else:
                self.notify(f"✗ Reset method '{method_name}' not found", severity="error")

        except Exception as e:
            self.notify(f"✗ Failed to reset {key}: {str(e)}", severity="error")

    def _update_table_row(self, key: str, value: Any) -> None:
        """Helper to update a specific row's value column visually."""
        table = self.query_one(DataTable)
        display_val = str(value) if value != "" else "(empty)"

        # 'key' was used as the row_key in add_row, so we can address it directly
        # The value column is at index 1 (0 is key, 1 is value)
        try:
            col_key = list(table.columns.keys())[1]
            table.update_cell(key, col_key, display_val)
        except Exception:
            # Fallback if key update fails, reload all
            self._load_table_data()


class ConfigTUICommand(SupportsCliCommand):
    """TUI command for config management."""

    command_string = "config-tui"
    help_string = "Launch interactive TUI for managing cognee configuration"
    docs_url = f"{DEFAULT_DOCS_URL}/usage/config-tui"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        try:
            # Import here to check if Textual is actually installed
            from textual.app import App

            class ConfigTUIApp(App):
                """Simple app wrapper for config TUI."""

                CSS = """
                Screen { background: $surface; }
                """

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
