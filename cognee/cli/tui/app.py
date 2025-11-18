"""
Cognee TUI - Main Application
Text-based User Interface for managing Cognee knowledge graphs
"""
from typing import ClassVar
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, Button, Markdown
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.events import Key
from cognee.cli.tui.screens.home import HomeScreen


class CogneeTUI(App):
    """Cognee Terminal User Interface Application"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    .box {
        border: solid $primary;
        background: $panel;
        padding: 1 2;
        margin: 1;
    }
    
    Button {
        margin: 1 2;
        min-width: 30;
    }
    
    Button:hover {
        background: $primary;
    }
    
    #menu-container {
        width: 60;
        height: auto;
        border: heavy $primary;
        background: $panel;
        padding: 2;
    }
    
    .title {
        text-align: center;
        text-style: bold;
        color: $accent;
        padding: 1;
    }
    
    .center {
        align: center middle;
    }

    /* Brand color for Manage Context button only */
    #context {
        background: #5C10F4;
        color: #F4F4F4;
        border: tall #5C10F4;
    }
    #context:hover {
        background: #A550FF;
        border: tall #A550FF;
    }
    """
    
    TITLE = "Cognee TUI - AI Memory and Context Manager"
    SUB_TITLE = "Navigate with arrow keys • Press ? for help"
    
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("?", "help", "Help"),
        Binding("d", "toggle_dark", "Toggle Dark Mode"),
    ]
    def on_mount(self) -> None:
        """Initialize the app with the home screen"""
        self.push_screen(HomeScreen())
    
    def action_help(self) -> None:
        """Show help information"""
        help_text = """
# Cognee TUI Help

## Navigation
- Arrow Keys: Navigate between UI elements
- Enter: Select/activate items
- Tab: Move to next field
- Esc: Go back

## Keyboard Shortcuts
- q: Quit application
- d: Toggle dark/light mode
- ?: Show this help

## Workflow
1. Add Context: Add data sources
2. Cognify: Process data
3. Search: Query knowledge graph
4. Settings: Configure API keys
"""
        self.push_screen(HelpScreen(help_text))



class HelpScreen(Screen):
    """Help screen"""
    
    def __init__(self, help_text: str):
        super().__init__()
        self.help_text = help_text
    
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Markdown(self.help_text)
        yield Button("Close (Esc)", id="close", variant="primary")
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.app.pop_screen()
    
    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.app.pop_screen()


def run_tui(mouse: bool = True) -> None:
    """Entry point to run the TUI application.
    
    Args:
        mouse: Enable mouse support (default: True)
    """
    app = CogneeTUI()
    app.run(mouse=mouse)
