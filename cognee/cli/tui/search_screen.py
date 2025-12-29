import asyncio
from textual.app import ComposeResult
from textual.widgets import Input, Label, Static, Select
from textual.containers import Container, Vertical, ScrollableContainer
from textual.binding import Binding
from cognee.cli.tui.base_screen import BaseTUIScreen


class SearchTUIScreen(BaseTUIScreen):
    """Simple search screen with query input and results display."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "search", "Search"),
    ]

    CSS = (
        BaseTUIScreen.CSS
        + """
    #search-form {
        height: auto;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    #search-form Label {
        margin-bottom: 0;
        color: $text-muted;
    }

    #search-form Input, #search-form Select {
        margin-bottom: 1;
    }

    #results-container {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #results-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #results-content {
        height: 1fr;
        overflow-y: auto;
    }
    """
    )

    def __init__(self):
        super().__init__()
        self.is_searching = False

    def compose_content(self) -> ComposeResult:
        with Container(classes="tui-main-container"):
            with Container(classes="tui-title-wrapper"):
                yield Static("🔍 Search Data", classes="tui-title-bordered")
            with Vertical(id="search-form"):
                yield Label("Query:", classes="tui-label-spaced")
                yield Input(placeholder="Enter your search query...", id="query-input")
                yield Label("Search Type:", classes="tui-label-spaced")
                yield Select(
                    [
                        ("Graph Completion (Recommended)", "GRAPH_COMPLETION"),
                        ("RAG Completion", "RAG_COMPLETION"),
                        ("Chunks", "CHUNKS"),
                        ("Summaries", "SUMMARIES"),
                        ("Coding Rules", "CODING_RULES"),
                    ],
                    value="GRAPH_COMPLETION",
                    id="query-type-select",
                )
            with Container(id="results-container"):
                yield Static("Results", id="results-title")
                with ScrollableContainer(id="results-content"):
                    yield Static(
                        "Enter a query and click Search to see results.", id="results-text"
                    )

    def compose_footer(self) -> ComposeResult:
        yield Static("Ctrl+S: Search  •  Esc: Back  •  q: Quit", classes="tui-footer")

    def on_mount(self) -> None:
        """Focus the query input on mount."""
        query_input = self.query_one("#query-input", Input)
        query_input.focus()

    def action_back(self) -> None:
        """Go back to home screen."""
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        """Quit the entire application."""
        self.app.exit()

    def action_search(self) -> None:
        """Trigger search action."""
        if not self.is_searching:
            self._perform_search()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in query input."""
        if event.input.id == "query-input":
            self._perform_search()

    def _perform_search(self) -> None:
        """Perform the search operation."""
        if self.is_searching:
            return

        query_input = self.query_one("#query-input", Input)
        query_text = query_input.value.strip()

        if not query_text:
            self.notify("Please enter a search query", severity="warning")
            return

        query_type_select = self.query_one("#query-type-select", Select)
        query_type = str(query_type_select.value)

        self.is_searching = True
        self.notify(f"Searching for: {query_text}", severity="information")

        # Update results to show loading
        results_text = self.query_one("#results-text", Static)
        results_text.update("🔍 Searching...")

        # Run async search
        asyncio.create_task(self._async_search(query_text, query_type))

    async def _async_search(self, query_text: str, query_type: str) -> None:
        """Async search operation."""
        try:
            import cognee
            from cognee.modules.search.types import SearchType

            # Convert string to SearchType enum
            search_type = SearchType[query_type]
            # Perform search
            results = await cognee.search(
                query_text=query_text,
                query_type=search_type,
                system_prompt_path="answer_simple_question.txt",
                top_k=10,
            )

            # Update results display
            results_text = self.query_one("#results-text", Static)

            if not results:
                results_text.update("No results found for your query.")
            else:
                # Format results based on type
                if query_type in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
                    formatted = "\n\n".join([f"📝 {result}" for result in results])
                elif query_type == "CHUNKS":
                    formatted = "\n\n".join(
                        [f"📄 Chunk {i + 1}:\n{result}" for i, result in enumerate(results)]
                    )
                else:
                    formatted = "\n\n".join([f"• {result}" for result in results])

                results_text.update(formatted)

            self.notify(f"✓ Found {len(results)} result(s)", severity="information")

        except Exception as e:
            results_text = self.query_one("#results-text", Static)
            results_text.update(f"❌ Error: {str(e)}")
            self.notify(f"Search failed: {str(e)}", severity="error")

        finally:
            self.is_searching = False
