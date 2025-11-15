"""Context Management Screen"""
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Button, Static
from textual.containers import Container
from textual.binding import Binding


class ContextScreen(Screen):
    """Context management screen"""
    
    BINDINGS = [Binding("escape", "back", "Back")]
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Static("[bold]📁 Context Management[/bold]\n", classes="title")
            yield Static("Context management features coming soon!")
            yield Button("← Back", id="back_btn")
        yield Footer()
    
    def on_button_pressed(self, event) -> None:
        self.app.pop_screen()
    
    def action_back(self) -> None:
        self.app.pop_screen()
