import asyncio
from uuid import UUID
from textual.app import ComposeResult
from textual.widgets import Input, Button, Static, Label
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding
from cognee.cli.tui.base_screen import BaseTUIScreen
from cognee.modules.data.methods.delete_dataset_by_name import delete_dataset_by_name
from cognee.modules.data.methods.delete_data_by_user import delete_data_by_user
from cognee.modules.users.methods import get_default_user


class DeleteTUIScreen(BaseTUIScreen):
    """Simple delete screen with input fields for dataset name, user ID, or delete all."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "delete", "Delete"),
        Binding("ctrl+a", "delete_all", "Delete All", priority=True),
    ]

    CSS = (
        BaseTUIScreen.CSS
        + """
    #button-group {
        height: auto;
        align: center middle;
        margin-top: 2;
    }
    """
    )

    def __init__(self):
        super().__init__()
        self.is_processing = False

    def compose_content(self) -> ComposeResult:
        with Container(classes="tui-main-container"):
            with Container(classes="tui-title-wrapper"):
                yield Static("🗑  Delete Data", classes="tui-title-bordered")
            with Vertical(id="delete-form", classes="tui-form"):
                with Vertical(classes="tui-input-group"):
                    yield Label("Dataset Name (optional):", classes="tui-label")
                    yield Input(
                        placeholder="Enter dataset name to delete specific dataset",
                        id="dataset-input",
                    )

                with Vertical(classes="tui-input-group"):
                    yield Label("User ID (optional):", classes="tui-label")
                    yield Input(
                        placeholder="Enter user ID to delete user's data or leave empty for default user.",
                        id="user-input",
                    )

                with Horizontal(id="button-group"):
                    yield Button("Delete", variant="error", id="delete-btn")
                    yield Button("Delete All", variant="error", id="delete-all-btn")

                yield Static("", classes="tui-status")

    def compose_footer(self) -> ComposeResult:
        yield Static(
            "Ctrl+s: Delete  •  Ctrl+a: Delete All  •  Esc: Back  •  q: Quit", classes="tui-footer"
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

    def action_delete(self) -> None:
        """Delete the dataset."""
        if not self.is_processing:
            self._handle_delete()

    def action_delete_all(self) -> None:
        """Delete all data."""
        if not self.is_processing:
            self._handle_delete_all()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if self.is_processing:
            return
        if event.button.id == "delete-btn":
            self._handle_delete()
        elif event.button.id == "delete-all-btn":
            self._handle_delete_all()
        elif event.button.id == "cancel-btn":
            self.app.pop_screen()

    def _handle_delete(self) -> None:
        status = self.query_one(".tui-status", Static)
        status.update("🔍 Starting the deletion process...")
        """Handle delete operation for dataset or user."""
        if self.is_processing:
            return

        dataset_input = self.query_one("#dataset-input", Input)
        user_input = self.query_one("#user-input", Input)

        dataset_name = dataset_input.value.strip() or None
        user_id = user_input.value.strip() or None

        if not dataset_name and not user_id:
            status.update("⚠️  Please enter a dataset name or user ID")
            return

        self.is_processing = True
        status.update("🔍 Checking data to delete...")
        # Run async delete operation
        asyncio.create_task(self._delete_async(dataset_name, user_id))

    async def _delete_async(self, dataset_name: str | None, user_id: str | None) -> None:
        """Async function to delete data."""
        status = self.query_one(".tui-status", Static)
        try:
            if dataset_name:
                if user_id is None:
                    user = await get_default_user()
                    resolved_user_id = user.id
                else:
                    resolved_user_id = UUID(user_id)
                await delete_dataset_by_name(dataset_name, resolved_user_id)
            else:
                await delete_data_by_user(resolved_user_id)
            status.update(f"✓ Successfully deleted dataset '{dataset_name}'.")
        except Exception as e:
            status.update(f"✗ Error: {str(e)}")
        finally:
            self.is_processing = False
            self.clear_input()

    def _handle_delete_all(self) -> None:
        """Handle delete all operation with confirmation."""
        if self.is_processing:
            return
        user_input = self.query_one("#user-input", Input)
        user_id = user_input.value.strip() or None

        def handle_confirm(confirmed: bool) -> None:
            if confirmed:
                asyncio.create_task(self._perform_delete_all(user_id))

        self.app.push_screen(DeleteAllConfirmModal(), handle_confirm)

    async def _perform_delete_all(self, user_id: str | None) -> None:
        """Perform the actual delete all operation."""
        status = self.query_one(".tui-status", Static)
        self.is_processing = True

        try:
            status.update("🔍 Deleting all data...")
            if user_id is None:
                user = await get_default_user() 
                resolved_user_id = user.id
            else:
                resolved_user_id = UUID(user_id)
            await delete_data_by_user(resolved_user_id)
            status.update(f"✓ Successfully deleted all data by user {resolved_user_id}")

            # Clear inputs
            dataset_input = self.query_one("#dataset-input", Input)
            user_input = self.query_one("#user-input", Input)
            dataset_input.value = ""
            user_input.value = ""

        except Exception as e:
            status.update(f"✗ Error: {str(e)}")
        finally:
            self.is_processing = False

    def clear_input(self) -> None:
        dataset_input = self.query_one("#dataset-input", Input)
        user_input = self.query_one("#user-input", Input)
        dataset_input.value = ""
        user_input.value = ""


class DeleteAllConfirmModal(BaseTUIScreen):
    """Modal screen for confirming delete all action."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = (
        BaseTUIScreen.CSS
        + """
    DeleteAllConfirmModal {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: 20;
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

    #confirm-warning {
        text-align: center;
        color: $warning;
        text-style: bold;
        margin-bottom: 2;
    }
    """
    )

    def compose_content(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label("⚠️  DELETE ALL DATA", id="confirm-title")
            yield Label("This will delete ALL data from cognee", classes="tui-dialog-message")
            yield Label("This operation is IRREVERSIBLE!", id="confirm-warning")
            with Horizontal(classes="tui-dialog-buttons"):
                yield Button("Delete All", variant="error", id="confirm-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
