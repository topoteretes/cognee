import sys
import asyncio
from unittest.mock import MagicMock, AsyncMock

# 1. Setup Mocks for cognee dependencies
mock_cognee = MagicMock()
sys.modules["cognee"] = mock_cognee
sys.modules["cognee.version"] = MagicMock()
sys.modules["cognee.cli.tui.common_styles"] = MagicMock()
sys.modules["cognee.cli.tui.common_styles"].COMMON_STYLES = ""

# Define the exception we want to catch
class EntityNotFoundError(Exception):
    pass

# Setup the exception in the mocked module structure
sys.modules["cognee.infrastructure"] = MagicMock()
sys.modules["cognee.infrastructure.databases"] = MagicMock()
sys.modules["cognee.infrastructure.databases.exceptions"] = MagicMock()
exceptions_mock = MagicMock()
exceptions_mock.EntityNotFoundError = EntityNotFoundError
sys.modules["cognee.infrastructure.databases.exceptions.exceptions"] = exceptions_mock

# Setup search types
sys.modules["cognee.modules"] = MagicMock()
sys.modules["cognee.modules.search"] = MagicMock()
types_mock = MagicMock()

# Mock SearchType to support item access
class MockSearchTypeMeta(type):
    def __getitem__(cls, key):
        return key
class MockSearchType(metaclass=MockSearchTypeMeta):
    pass
types_mock.SearchType = MockSearchType
sys.modules["cognee.modules.search.types"] = types_mock

import importlib.util
import os

# ... existing mocks ...
# Ensure we have deep mocks for structure
sys.modules["cognee.cli"] = MagicMock()
sys.modules["cognee.cli.tui"] = MagicMock()

# Mock BaseTUIScreen specifically
base_screen_mock = MagicMock()
class MockBaseScreen:
    CSS = ""
    def __init__(self):
        pass
    def compose_header(self): yield from ()
    def compose_footer(self): yield from ()
base_screen_mock.BaseTUIScreen = MockBaseScreen
# Crucial: Mock the specific module path search_screen tries to import from
sys.modules["cognee.cli.tui.base_screen"] = base_screen_mock

# Also mock textual.binding which is imported at top level
sys.modules["textual.binding"] = MagicMock()

# Now load the file directly
module_path = os.path.join(os.getcwd(), "cognee/cli/tui/search_screen.py")
spec = importlib.util.spec_from_file_location("search_screen_mod", module_path)
search_screen_mod = importlib.util.module_from_spec(spec)

# Before executing, ensure imports in that file will resolve to our mocks
# The file does: from textual...
# Real Textual is explicitly NOT mocked in sys.modules so it loads real textual (if installed)
# But we mocked textual.binding above? 
# Actually, let's NOT mock textual.binding if we can avoid it, or mock it if it's simple.
# Real code: from textual.binding import Binding. 
# If textual is installed, we should leverage it. If not, mock it.
try:
    import textual
except ImportError:
    # If textual not installed/available in this step runner, we must mock it all
    sys.modules["textual"] = MagicMock()
    sys.modules["textual.app"] = MagicMock()
    sys.modules["textual.widgets"] = MagicMock()
    sys.modules["textual.containers"] = MagicMock()
    sys.modules["textual.binding"] = MagicMock()
    # We need to provide Widget classes that search_screen inherits/uses
    # It imports: Input, Label, Static, Select, ListView, ListItem
    # It uses: ComposeResult (type)
    
    # Simple Mock widgets
    class MockWidget:
        def __init__(self, *args, **kwargs): pass
        def focus(self): pass
        def mount(self, *args): pass
        def clear(self): pass
        def update(self, *args): pass
    
    sys.modules["textual.widgets"].Input = MockWidget
    sys.modules["textual.widgets"].Label = MockWidget
    sys.modules["textual.widgets"].Static = MockWidget
    sys.modules["textual.widgets"].Select = MockWidget
    sys.modules["textual.widgets"].ListView = MockWidget
    sys.modules["textual.widgets"].ListItem = MockWidget
    
    sys.modules["textual.containers"].Container = MagicMock()
    sys.modules["textual.containers"].Vertical = MagicMock()
    sys.modules["textual.app"].ComposeResult = MagicMock()

# Execute the module
spec.loader.exec_module(search_screen_mod)
SearchTUIScreen = search_screen_mod.SearchTUIScreen

async def test_empty_graph_handling():
    print("Testing Empty Graph (EntityNotFoundError) Handling...")
    
    # Instantiate screen
    screen = SearchTUIScreen()
    
    # Mock query_one to return our list view mock
    results_list_mock = MagicMock()
    
    def query_one_side_effect(selector, type_cls=None):
        if "ListView" in str(type_cls) or "list" in str(selector):
            return results_list_mock
        return MagicMock() # For other queries like Static
        
    screen.query_one = MagicMock(side_effect=query_one_side_effect)
    screen.notify = MagicMock()
    
    # Configure cognee.search to raise EntityNotFoundError
    mock_cognee.search = AsyncMock(side_effect=EntityNotFoundError("Graph is empty"))
    
    # Run the method
    # Note: query_type needs to be a valid key in our MockSearchType or just a string if we mocked it right
    await screen._async_search("test query", "GRAPH_COMPLETION")
    
    # Verification
    # 1. usage of clear()
    results_list_mock.clear.assert_called()
    
    # 2. usage of mount() with correct message
    assert results_list_mock.mount.called
    
    # Check that notify was called with warning
    # We allow flexible matching for the exact message but check 'warning' severity
    assert screen.notify.called, "notify was not called"
    args = screen.notify.call_args
    assert args is not None
    # Assert specific conditions on args
    assert args[1].get('severity') == 'warning' or 'Knowledge graph is empty' in args[0][0]
        
    print("SUCCESS: EntityNotFoundError was caught and handled correctly.")

async def test_generic_error_handling():
    print("\nTesting Generic Error Handling...")
    screen = SearchTUIScreen()
    results_list_mock = MagicMock()
    screen.query_one = MagicMock(return_value=results_list_mock)
    screen.notify = MagicMock()
    
    # Configure generic error
    mock_cognee.search = AsyncMock(side_effect=Exception("Something bad happened"))
    
    await screen._async_search("test query", "GRAPH_COMPLETION")
    
    screen.notify.assert_called()
    args = screen.notify.call_args
    # Check for error severity or message
    assert args[1].get('severity') == 'error' or "Search failed" in args[0][0]
    print("SUCCESS: Generic Exception was caught and handled correctly.")

async def test_success_path():
    print("\nTesting Success Path...")
    screen = SearchTUIScreen()
    results_list_mock = MagicMock()
    screen.query_one = MagicMock(return_value=results_list_mock)
    screen.notify = MagicMock()
    
    # Configure success
    mock_cognee.search = AsyncMock(return_value=["Result 1", "Result 2"])
    
    await screen._async_search("test query", "GRAPH_COMPLETION")
    
    assert results_list_mock.clear.called
    assert results_list_mock.mount.called
    # We expect 2 mount calls for results
    assert results_list_mock.mount.call_count == 2
    print("SUCCESS: Search results were displayed.")

if __name__ == "__main__":
    asyncio.run(test_empty_graph_handling())
    asyncio.run(test_generic_error_handling())
    asyncio.run(test_success_path())
