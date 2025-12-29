from textual.app import ComposeResult
from textual.widgets import ListView, ListItem, Static
from textual.containers import Container, Horizontal
from textual.binding import Binding

from cognee.cli.tui.base_screen import BaseTUIScreen
from cognee.cli.tui.config_screen import ConfigTUIScreen
from cognee.cli.tui.add_screen import AddTUIScreen
from cognee.cli.tui.cognify_screen import CognifyTUIScreen
from cognee.cli.tui.search_screen import SearchTUIScreen
from cognee.cli.tui.delete_screen import DeleteTUIScreen


def make_item(icon: str, command: str, description: str) -> ListItem:
    """Compose a ListItem that contains a Horizontal container with 3 children."""
    return ListItem(
        Horizontal(
            Static(icon, classes="cmd-icon"),
            Static(command, classes="cmd-name"),
            Static(description, classes="cmd-desc"),
            classes="cmd-row",
        )
    )


class HomeScreen(BaseTUIScreen):
    """Home screen with command selection menu."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "quit_app", "Quit"),
        Binding("enter", "select", "Select"),
        Binding("up", "nav_up", "Up", priority=True),
        Binding("down", "nav_down", "Down", priority=True),
    ]

    CSS = (
        BaseTUIScreen.CSS
        + """
    ListView > ListItem {
        width: 100%;
        padding: 0;
        margin: 0;
    }
    
    .menu-list > ListItem {
        width: 100%;
        padding: 0;
        margin: 0;
    }

    .menu-list {
        height: auto;
        background: $surface;
        border: none;
        padding: 0 0;
    }

    ListView {
        height: auto;
        background: $surface;
        border: none;
        padding: 0 0;
    }

    ListItem {
        background: $surface;
        color: $text;
        width: 100%;
        height: 3;
    }
    
    ListItem:focus {
        outline: none;
    }

    ListItem.highlighted {
        background: $primary-darken-3;
        color: $text;
    }
    ListItem.highlighted .cmd-name {
        text-style: bold;
        color: $accent;
    }

    .cmd-row {
        width: 100%;
        height: auto;
        align-horizontal: left;
        align-vertical: middle;
        height: 1fr;
    }

    .cmd-icon {
        width: 4;
        text-align: center;
        color: $text-muted;
    }

    .cmd-name {
        width: 14;
        padding-left: 1;
        text-style: bold;
    }

    .cmd-desc {
        width: 1fr;
        overflow: auto;
        padding-left: 1;
        color: $text-muted;
    }
    """
    )

    def __init__(self):
        super().__init__()
        self.lv = None
        self.current_index = 0

    def compose_content(self) -> ComposeResult:
        with Container(classes="tui-main-container"):
            with Container(classes="tui-title-wrapper"):
                yield Static("Select Command", classes="tui-title-bordered")
            with Container(classes="tui-bordered-wrapper"):
                yield ListView(
                    make_item("📥", "add", "Add data to cognee"),
                    make_item("🔍", "search", "Search data in cognee"),
                    make_item("⚡", "cognify", "Process data in cognee"),
                    make_item("🗑️", "delete", "Delete data from cognee"),
                    make_item("⚙️", "config", "Configure cognee settings"),
                    id="menu-list",
                    classes="menu-list",
                )

    def compose_footer(self) -> ComposeResult:
        yield Static("↑↓: Navigate  •  Enter: Select  •  q/Esc: Quit", classes="tui-footer")

    def on_mount(self) -> None:
        """Focus the list view on mount."""
        self.lv = self.query_one(ListView)
        self.current_index = 0
        self.set_focus(self.lv)
        self._apply_highlight()

    def _apply_highlight(self) -> None:
        lv = self.lv
        children = list(lv.children)
        self.lv.index = self.current_index
        for idx, item in enumerate(children):
            if idx == self.current_index:
                item.add_class("highlighted")
            else:
                item.remove_class("highlighted")

    def action_nav_up(self) -> None:
        self.current_index = max(0, self.current_index - 1)
        self._apply_highlight()

    def action_nav_down(self) -> None:
        children = list(self.lv.children)
        self.current_index = min(len(children) - 1, self.current_index + 1)
        self._apply_highlight()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        selected_index = event.index
        self.current_index = selected_index
        self._apply_highlight()
        if selected_index == 0:  # add
            self.app.push_screen(AddTUIScreen())
        elif selected_index == 1:  # search
            self.app.push_screen(SearchTUIScreen())
        elif selected_index == 2:  # cognify
            self.app.push_screen(CognifyTUIScreen())
        elif selected_index == 3:  # delete
            self.app.push_screen(DeleteTUIScreen())
        elif selected_index == 4:  # config
            self.app.push_screen(ConfigTUIScreen())
        else:
            self.app.exit()

    def action_select(self) -> None:
        """Select the current item."""
        list_view = self.query_one(ListView)
        list_view.action_select_cursor()

    def action_quit_app(self) -> None:
        """Quit the entire application."""
        self.app.exit()
