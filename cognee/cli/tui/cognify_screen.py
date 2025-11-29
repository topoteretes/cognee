import asyncio
from textual.app import ComposeResult
from textual.widgets import Input, Label, Button, Static, Checkbox, RadioSet, RadioButton
from textual.containers import Container, Vertical
from textual.binding import Binding

from cognee.cli.tui.base_screen import BaseTUIScreen
from cognee.cli.config import CHUNKER_CHOICES


class CognifyTUIScreen(BaseTUIScreen):
    """TUI screen for cognifying data in cognee."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "submit", "Submit"),
    ]

    CSS = BaseTUIScreen.CSS + """
    CognifyTUIScreen {
        background: $surface;
    }

    #cognify-container {
        height: auto;
        padding: 1;
        content-align: center middle;
    }

    #cognify-form {
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
        margin-bottom: 0;
    }

    Input {
        width: 100%;
        margin-bottom: 1;
    }

    Checkbox {
        margin-top: 1;
        margin-bottom: 1;
    }

    RadioSet {
        margin-top: 0;
        margin-bottom: 1;
        height: auto;
    }

    RadioButton {
        height: 1;
    }

    #status-message {
        margin-top: 2;
        text-align: center;
        height: auto;
    }

    #cognify-footer {
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
        with Container(id="cognify-container"):
            yield Label("Cognify Data", id="form-title")
            with Vertical(id="cognify-form"):
                yield Label("Dataset Name (optional, leave empty for all):", classes="field-label")
                yield Input(
                    placeholder="Leave empty to process all datasets",
                    value="",
                    id="dataset-input"
                )
                
                yield Label("Chunker Type:", classes="field-label")
                with RadioSet(id="chunker-radio"):
                    for chunker in CHUNKER_CHOICES:
                        yield RadioButton(chunker, value=(chunker == "TextChunker"))
                
                yield Checkbox("Run in background", id="background-checkbox")
            yield Static("", id="status-message")

    def compose_footer(self) -> ComposeResult:
        yield Static(
            "Ctrl+S: Start  •  Esc: Back  •  q: Quit",
            id="cognify-footer"
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

    def action_submit(self) -> None:
        """Submit the form."""
        if not self.is_processing:
            self._submit_cognify()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "submit-btn" and not self.is_processing:
            self._submit_cognify()

    def _submit_cognify(self) -> None:
        """Process and submit the cognify request."""
        dataset_input = self.query_one("#dataset-input", Input)
        chunker_radio = self.query_one("#chunker-radio", RadioSet)
        background_checkbox = self.query_one("#background-checkbox", Checkbox)
        status = self.query_one("#status-message", Static)

        dataset_name = dataset_input.value.strip() or None
        chunker_type = str(chunker_radio.pressed_button.label) if chunker_radio.pressed_button else "TextChunker"
        run_background = background_checkbox.value

        self.is_processing = True
        status.update("[yellow]⏳ Starting cognification...[/yellow]")
        
        # Disable inputs during processing
        dataset_input.disabled = True
        chunker_radio.disabled = True
        background_checkbox.disabled = True
        self.query_one("#submit-btn", Button).disabled = True

        # Run async cognify operation
        asyncio.create_task(self._cognify_async(dataset_name, chunker_type, run_background))

    async def _cognify_async(self, dataset_name: str | None, chunker_type: str, run_background: bool) -> None:
        """Async function to cognify data."""
        status = self.query_one("#status-message", Static)
        
        try:
            import cognee
            from cognee.modules.chunking.TextChunker import TextChunker
            
            # Get chunker class
            chunker_class = TextChunker
            if chunker_type == "LangchainChunker":
                try:
                    from cognee.modules.chunking.LangchainChunker import LangchainChunker
                    chunker_class = LangchainChunker
                except ImportError:
                    status.update("[yellow]⚠ LangchainChunker not available, using TextChunker[/yellow]")
            elif chunker_type == "CsvChunker":
                try:
                    from cognee.modules.chunking.CsvChunker import CsvChunker
                    chunker_class = CsvChunker
                except ImportError:
                    status.update("[yellow]⚠ CsvChunker not available, using TextChunker[/yellow]")
            
            # Prepare datasets parameter
            datasets = [dataset_name] if dataset_name else None
            
            await cognee.cognify(
                datasets=datasets,
                chunker=chunker_class,
                run_in_background=run_background,
            )
            
            if run_background:
                status.update("[green]✓ Cognification started in background![/green]")
            else:
                status.update("[green]✓ Cognification completed successfully![/green]")
            
        except Exception as e:
            status.update(f"[red]✗ Failed to cognify: {str(e)}[/red]")
        
        finally:
            # Re-enable inputs
            self.is_processing = False
            dataset_input = self.query_one("#dataset-input", Input)
            chunker_radio = self.query_one("#chunker-radio", RadioSet)
            background_checkbox = self.query_one("#background-checkbox", Checkbox)
            dataset_input.disabled = False
            chunker_radio.disabled = False
            background_checkbox.disabled = False
            self.query_one("#submit-btn", Button).disabled = False
            dataset_input.focus()