import asyncio
from textual.app import ComposeResult
from textual.widgets import Input, Label, Button, Static, TextArea
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding

from cognee.cli.tui.base_screen import BaseTUIScreen


class AddTUIScreen(BaseTUIScreen):
    """TUI screen for adding data to cognee."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "submit", "Submit"),
        Binding("ctrl+v", "paste", "Paste", show=False),
    ]

    CSS = BaseTUIScreen.CSS + """
    AddTUIScreen {
        background: $surface;
    }

    #add-container {
        height: auto;
        padding: 1;
        content-align: center middle;
    }

    #add-form {
        width: 100%;
        height: auto;
        border: solid $primary;
        padding: 2;
        background: $surface;
    }

    #form-title {
        text-align: center;
        width: 100%;
        text-style: bold;
        color: $accent;
        margin-bottom: 2;
    }

    .field-label {
        color: $text-muted;
        margin-top: 1;
        margin-bottom: 1;
    }

    Input {
        width: 100%;
        margin-bottom: 1;
    }

    #data-input {
        height: 8;
        min-height: 8;
    }

    #status-message {
        margin-top: 2;
        text-align: center;
        height: auto;
    }

    #add-footer {
        dock: bottom;
        padding: 1 0;
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
        with Container(id="add-container"):
            yield Label("Add Data to Cognee", id="form-title")
            with Vertical(id="add-form"):
                yield Label("Data (text, file path, URL, or S3 path):", classes="field-label")
                yield TextArea(
                    "",
                    id="data-input",
                )
                
                yield Label("Dataset Name:", classes="field-label")
                yield Input(
                    placeholder="main_dataset",
                    value="main_dataset",
                    id="dataset-input"
                )
            yield Static("", id="status-message")

    def compose_footer(self) -> ComposeResult:
        yield Static(
            "Ctrl+S: Add  •  Esc: Back  •  q: Quit",
            id="add-footer"
        )

    def on_mount(self) -> None:
        """Focus the data input on mount."""
        data_input = self.query_one("#data-input", TextArea)
        data_input.focus()

    def action_back(self) -> None:
        """Go back to home screen."""
        if not self.is_processing:
            self.app.pop_screen()

    def action_quit_app(self) -> None:
        """Quit the entire application."""
        self.app.exit()

    def action_paste(self) -> None:
        """Handle paste action - Textual handles this automatically."""
        pass

    def action_submit(self) -> None:
        """Submit the form."""
        if not self.is_processing:
            self._submit_data()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "submit-btn" and not self.is_processing:
            self._submit_data()

    def _submit_data(self) -> None:
        """Process and submit the data."""
        data_input = self.query_one("#data-input", TextArea)
        dataset_input = self.query_one("#dataset-input", Input)
        status = self.query_one("#status-message", Static)

        data = data_input.text.strip()
        dataset_name = dataset_input.value.strip() or "main_dataset"

        if not data:
            status.update("[red]✗ Please enter data to add[/red]")
            return

        self.is_processing = True
        status.update("[yellow]⏳ Processing...[/yellow]")
        
        # Disable inputs during processing
        data_input.disabled = True
        dataset_input.disabled = True
        self.query_one("#submit-btn", Button).disabled = True

        # Run async add operation
        asyncio.create_task(self._add_data_async(data, dataset_name))

    async def _add_data_async(self, data: str, dataset_name: str) -> None:
        """Async function to add data to cognee."""
        status = self.query_one("#status-message", Static)
        
        try:
            import cognee
            
            await cognee.add(data=data, dataset_name=dataset_name)
            
            status.update(f"[green]✓ Successfully added data to dataset '{dataset_name}'[/green]")
            
            # Clear the data input after successful add
            data_input = self.query_one("#data-input", TextArea)
            data_input.clear()
            
        except Exception as e:
            status.update(f"[red]✗ Failed to add data: {str(e)}[/red]")
        
        finally:
            # Re-enable inputs
            self.is_processing = False
            data_input = self.query_one("#data-input", TextArea)
            dataset_input = self.query_one("#dataset-input", Input)
            data_input.disabled = False
            dataset_input.disabled = False
            self.query_one("#submit-btn", Button).disabled = False
            data_input.focus()
