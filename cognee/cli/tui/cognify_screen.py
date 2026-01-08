import asyncio
from textual.app import ComposeResult
from textual.widgets import Input, Label, Static, Checkbox, RadioSet, RadioButton
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

    CSS = (
        BaseTUIScreen.CSS
        + """
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
    """
    )

    def __init__(self):
        super().__init__()
        self.is_processing = False

    def compose_content(self) -> ComposeResult:
        with Container(classes="tui-main-container"):
            with Container(classes="tui-title-wrapper"):
                yield Static("⚡ Cognify Data", classes="tui-title-bordered")
            with Vertical(classes="tui-form"):
                yield Label(
                    "Dataset Name:", classes="tui-label-spaced"
                )
                yield Input(
                    placeholder="Enter the dataset name here.", value="", id="dataset-input"
                )

                yield Label("Chunker Type:", classes="tui-label-spaced")
                with RadioSet(id="chunker-radio"):
                    for chunker in CHUNKER_CHOICES:
                        yield RadioButton(chunker, value=(chunker == "TextChunker"))

                yield Checkbox("Run in background", id="background-checkbox")
            yield Static("", classes="tui-status")

    def compose_footer(self) -> ComposeResult:
        yield Static("Ctrl+S: Start  •  Esc: Back  •  q: Quit", classes="tui-footer")

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

    def _submit_cognify(self) -> None:
        """Process and submit the cognify request."""
        dataset_input = self.query_one("#dataset-input", Input)
        chunker_radio = self.query_one("#chunker-radio", RadioSet)
        background_checkbox = self.query_one("#background-checkbox", Checkbox)
        status = self.query_one(".tui-status", Static)

        dataset_name = dataset_input.value.strip() or None
        chunker_type = (
            str(chunker_radio.pressed_button.label)
            if chunker_radio.pressed_button
            else "TextChunker"
        )
        run_background = background_checkbox.value

        self.is_processing = True
        status.update("[yellow]⏳ Starting cognification...[/yellow]")

        # Disable inputs during processing
        dataset_input.disabled = True
        chunker_radio.disabled = True
        background_checkbox.disabled = True

        # Run async cognify operation
        asyncio.create_task(self._cognify_async(dataset_name, chunker_type, run_background))

    async def _cognify_async(
        self, dataset_name: str | None, chunker_type: str, run_background: bool
    ) -> None:
        """Async function to cognify data."""
        status = self.query_one(".tui-status", Static)
        from cognee.modules.chunking.TextChunker import TextChunker

        try:
            # Get chunker class
            chunker_class = TextChunker
            if chunker_type == "LangchainChunker":
                try:
                    from cognee.modules.chunking.LangchainChunker import LangchainChunker
                except ImportError:
                    LangchainChunker = None
                if LangchainChunker is not None:
                    chunker_class = LangchainChunker
                else:
                    status.update(
                        "[yellow]⚠ LangchainChunker not available, using TextChunker[/yellow]"
                    )
            elif chunker_type == "CsvChunker":
                try:
                    from cognee.modules.chunking.CsvChunker import CsvChunker
                except ImportError:
                    CsvChunker = None
                if CsvChunker is not None:
                    chunker_class = CsvChunker
                else:
                    status.update("[yellow]⚠ CsvChunker not available, using TextChunker[/yellow]")

            # Prepare datasets parameter
            datasets = [dataset_name] if dataset_name else None
            import cognee

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
            dataset_input.focus()
