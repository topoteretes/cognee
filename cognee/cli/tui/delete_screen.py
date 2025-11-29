import asyncio
from textual.app import ComposeResult
from textual.widgets import Input, Button, Static, Label
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding

from cognee.cli.tui.base_screen import BaseTUIScreen
from cognee.modules.data.methods.get_deletion_counts import get_deletion_counts


class DeleteTUIScreen(BaseTUIScreen):
    """Simple delete screen with input fields for dataset name, user ID, or delete all."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("ctrl+d", "delete_all", "Delete All"),
    ]

    CSS = BaseTUIScreen.CSS + """
    DeleteTUIScreen {
        background: $surface;
    }

    #delete-container {
        height: auto;
        padding: 2;
        align: center top;
    }

    #delete-form {
        width: 80;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 2;
    }

    #form-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 2;
    }

    .input-group {
        height: auto;
        margin-bottom: 2;
    }

    .input-label {
        color: $text-muted;
        margin-bottom: 1;
    }

    Input {
        width: 100%;
        margin-bottom: 1;
    }

    #button-group {
        height: auto;
        align: center middle;
        margin-top: 2;
    }

    Button {
        margin: 0 1;
    }

    #status-message {
        text-align: center;
        margin-top: 2;
        height: auto;
    }

    #delete-footer {
        dock: bottom;
        height: 3;
        background: $boost;
        color: $text-muted;
        content-align: center middle;
        border: solid $primary;
    }
    """

    def __init__(self):
        super().__init__()
        self.is_processing = False

    def compose_content(self) -> ComposeResult:
        with Container(id="delete-container"):
            with Vertical(id="delete-form"):
                yield Label("🗑️  Delete Data", id="form-title")
                
                with Vertical(classes="input-group"):
                    yield Label("Dataset Name (optional):", classes="input-label")
                    yield Input(
                        placeholder="Enter dataset name to delete specific dataset",
                        id="dataset-input"
                    )
                
                with Vertical(classes="input-group"):
                    yield Label("User ID (optional):", classes="input-label")
                    yield Input(
                        placeholder="Enter user ID to delete user's data",
                        id="user-input"
                    )
                
                with Horizontal(id="button-group"):
                    yield Button("Delete", variant="error", id="delete-btn")
                    yield Button("Delete All", variant="error", id="delete-all-btn")
                    yield Button("Cancel", variant="default", id="cancel-btn")
                
                yield Static("", id="status-message")

    def compose_footer(self) -> ComposeResult:
        yield Static(
            "Enter dataset/user  •  Click Delete  •  Ctrl+D: Delete All  •  Esc: Back  •  q: Quit",
            id="delete-footer"
        )

    def on_mount(self) -> None:
        """Focus the dataset input on mount."""
        dataset_input = self.query_one("#dataset-input", Input)
        dataset_input.focus()

    def action_back(self) -> None:
        """Go back to home screen."""
        if not self.is_processing:
            self.app.pop_screen()

    def action_quit_app(self) -> None:
        """Quit the entire application."""
        self.app.exit()

    def action_delete_all(self) -> None:
        """Trigger delete all action."""
        if not self.is_processing:
            self._handle_delete_all()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if self.is_processing:
            return

        if event.button.id == "delete-btn":
            await self._handle_delete()
        elif event.button.id == "delete-all-btn":
            self._handle_delete_all()
        elif event.button.id == "cancel-btn":
            self.app.pop_screen()

    async def _handle_delete(self) -> None:
        """Handle delete operation for dataset or user."""
        if self.is_processing:
            return

        dataset_input = self.query_one("#dataset-input", Input)
        user_input = self.query_one("#user-input", Input)
        status = self.query_one("#status-message", Static)

        dataset_name = dataset_input.value.strip() or None
        user_id = user_input.value.strip() or None

        if not dataset_name and not user_id:
            status.update("⚠️  Please enter a dataset name or user ID")
            return

        self.is_processing = True
        status.update("🔍 Checking data to delete...")

        try:
            # Get preview of what will be deleted
            preview_data = await get_deletion_counts(
                dataset_name=dataset_name,
                user_id=user_id,
                all_data=False,
            )

            if not preview_data:
                status.update("✓ No data found to delete")
                self.is_processing = False
                return

            # Show preview and confirm
            preview_msg = (
                f"About to delete:\n"
                f"Datasets: {preview_data.datasets}\n"
                f"Entries: {preview_data.entries}\n"
                f"Users: {preview_data.users}"
            )
            status.update(preview_msg)

            # Perform deletion
            import cognee
            await cognee.delete(dataset_name=dataset_name, user_id=user_id)

            operation = f"dataset '{dataset_name}'" if dataset_name else f"data for user '{user_id}'"
            status.update(f"✓ Successfully deleted {operation}")

            # Clear inputs
            dataset_input.value = ""
            user_input.value = ""

        except Exception as e:
            status.update(f"✗ Error: {str(e)}")
        finally:
            self.is_processing = False

    def _handle_delete_all(self) -> None:
        """Handle delete all operation with confirmation."""
        if self.is_processing:
            return

        def handle_confirm(confirmed: bool) -> None:
            if confirmed:
                self.run_worker(self._perform_delete_all())

        self.app.push_screen(DeleteAllConfirmModal(), handle_confirm)

    async def _perform_delete_all(self) -> None:
        """Perform the actual delete all operation."""
        status = self.query_one("#status-message", Static)
        self.is_processing = True

        try:
            status.update("🔍 Checking all data...")

            # Get preview
            preview_data = await get_deletion_counts(
                dataset_name=None,
                user_id=None,
                all_data=True,
            )

            if not preview_data:
                status.update("✓ No data found to delete")
                self.is_processing = False
                return

            preview_msg = (
                f"Deleting ALL data:\n"
                f"Datasets: {preview_data.datasets}\n"
                f"Entries: {preview_data.entries}\n"
                f"Users: {preview_data.users}"
            )
            status.update(preview_msg)

            # Perform deletion
            import cognee
            await cognee.delete(dataset_name=None, user_id=None)

            status.update("✓ Successfully deleted all data")

            # Clear inputs
            dataset_input = self.query_one("#dataset-input", Input)
            user_input = self.query_one("#user-input", Input)
            dataset_input.value = ""
            user_input.value = ""

        except Exception as e:
            status.update(f"✗ Error: {str(e)}")
        finally:
            self.is_processing = False


class DeleteAllConfirmModal(BaseTUIScreen):
    """Modal screen for confirming delete all action."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = BaseTUIScreen.CSS + """
    DeleteAllConfirmModal {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: 13;
        border: thick $error;
        background: $surface;
        padding: 2;
    }

    #confirm-title {
        text-align: center;
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }

    #confirm-message {
        text-align: center;
        margin-bottom: 1;
    }

    #confirm-warning {
        text-align: center;
        color: $warning;
        text-style: bold;
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

    def compose_content(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label("⚠️  DELETE ALL DATA", id="confirm-title")
            yield Label("This will delete ALL data from cognee", id="confirm-message")
            yield Label("This operation is IRREVERSIBLE!", id="confirm-warning")
            with Horizontal(id="confirm-buttons"):
                yield Button("Delete All", variant="error", id="confirm-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)