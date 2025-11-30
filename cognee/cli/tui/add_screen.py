import asyncio
from textual.app import ComposeResult
from textual.widgets import Input, Label, Static, TextArea
from textual.containers import Container, Vertical
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
    #data-input {
        height: 8;
        min-height: 8;
    }
    """

    def __init__(self):
        super().__init__()
        self.is_processing = False

    def compose_content(self) -> ComposeResult:
        with Container(classes="tui-main-container"):
            with Container(classes="tui-title-wrapper"):
                yield Static("📥 Add Data to Cognee", classes="tui-title-bordered")
            with Vertical(classes="tui-form"):
                yield Label("Data (text, file path (/path/to/doc), URL, or S3 path (s3://bucket)):", classes="tui-label-spaced")
                yield TextArea(
                    "",
                    id="data-input",
                )
                
                yield Label("Dataset Name:", classes="tui-label-spaced")
                yield Input(
                    placeholder="main_dataset",
                    value="main_dataset",
                    id="dataset-input"
                )
            yield Static("", classes="tui-status")

    def compose_footer(self) -> ComposeResult:
        yield Static(
            "Ctrl+S: Add  •  Esc: Back  •  q: Quit",
            classes="tui-footer"
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

    def _submit_data(self) -> None:
        """Process and submit the data."""
        data_input = self.query_one("#data-input", TextArea)
        dataset_input = self.query_one("#dataset-input", Input)
        status = self.query_one(".tui-status", Static)

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

        # Run async add operation
        asyncio.create_task(self._add_data_async(data, dataset_name))

    async def _add_data_async(self, data: str, dataset_name: str) -> None:
        """Async function to add data to cognee."""
        status = self.query_one(".tui-status", Static)
        
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
            data_input.focus()
