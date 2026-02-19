import pytest
from unittest.mock import patch

from cognee.modules.observability.get_observe import get_observe
from cognee.modules.observability.observers import Observer
from cognee.modules.observability.exceptions import UnsupportedObserverError


def test_get_observe_raises_for_unsupported_observer():
    """Unsupported observer (e.g. LLMLITE, LANGSMITH) raises UnsupportedObserverError."""
    with patch("cognee.modules.observability.get_observe.get_base_config") as get_config:
        get_config.return_value = type("Config", (), {"monitoring_tool": Observer.LLMLITE})()

        with pytest.raises(UnsupportedObserverError, match="Unsupported observer"):
            get_observe()


def test_get_observe_raises_for_langsmith():
    """Observer.LANGSMITH raises UnsupportedObserverError."""
    with patch("cognee.modules.observability.get_observe.get_base_config") as get_config:
        get_config.return_value = type("Config", (), {"monitoring_tool": Observer.LANGSMITH})()

        with pytest.raises(UnsupportedObserverError, match="Unsupported observer"):
            get_observe()
