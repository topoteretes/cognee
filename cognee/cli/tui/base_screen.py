from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static

from cognee.version import get_cognee_version


class BaseTUIScreen(Screen):
    """Base screen class with constant header for all TUI screens."""

    # Subclasses should override this CSS and add their own styles
    CSS = """
    #header {
        dock: top;
        background: $boost;
        color: $text;
        content-align: center middle;
        border: solid $primary;
        text-style: bold;
        padding: 1;
    }
    """

    def compose_header(self) -> ComposeResult:
        """Compose the constant header widget."""
        version = get_cognee_version()
        yield Static(f"🧠 cognee v{version}", id="header")

    def compose_content(self) -> ComposeResult:
        """Override this method in subclasses to provide screen-specific content."""
        yield from ()

    def compose_footer(self) -> ComposeResult:
        """Override this method in subclasses to provide screen-specific footer."""
        yield from ()

    def compose(self) -> ComposeResult:
        """Compose the screen with header, content, and footer."""
        yield from self.compose_header()
        yield from self.compose_content()
        yield from self.compose_footer()