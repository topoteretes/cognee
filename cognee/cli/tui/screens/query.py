"""Query Screen"""

import asyncio
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Button, Static, Input, Markdown
from textual.containers import Container, Vertical, VerticalScroll
from textual.binding import Binding
import cognee


class QueryScreen(Screen):
    """Query screen"""

    BINDINGS = [Binding("escape", "back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Static("[bold]🔍 Search & Query[/bold]\n", classes="title")
            with Vertical():
                yield Static("Enter your question and run a graph-aware search:", classes="center")
                yield Input(placeholder="e.g., What are the main topics?", id="query_input")
                yield Button("Run Search", id="run_btn", variant="primary")
            with VerticalScroll():
                yield Markdown("", id="results_md")
            yield Button("← Back", id="back_btn")
        yield Footer()

    async def _set_results(self, content: str) -> None:
        try:
            md = self.query_one("#results_md", Markdown)
            md.update(content)
        except Exception:
            pass

    async def _run_search(self) -> None:
        query_input = self.query_one("#query_input", Input)
        query_text = (query_input.value or "").strip()
        if not query_text:
            await self._set_results(":warning: Please enter a question to search.")
            return
        try:
            await self._set_results("_Searching..._")
            results = await cognee.search(query_text=query_text)
            # Normalize results for display
            if isinstance(results, list):
                rendered = "\n".join(f"- {str(item)}" for item in results)
            else:
                rendered = str(results)
            await self._set_results(f"### Results\n\n{rendered}")
        except Exception as ex:
            await self._set_results(f"**Search failed:** {ex}")

    def on_button_pressed(self, event) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
            return
        if event.button.id == "run_btn":
            asyncio.create_task(self._run_search())
            return

    def action_back(self) -> None:
        self.app.pop_screen()
