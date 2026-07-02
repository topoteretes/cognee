import pytest
from unittest.mock import patch, MagicMock

from cognee.modules.observability.get_observe import get_observe
from cognee.modules.observability.observers import Observer
from cognee.modules.observability.exceptions import UnsupportedObserverError


def test_get_observe_raises_for_unsupported_observer():
    """Unsupported observer (e.g. LLMLITE) raises UnsupportedObserverError."""
    with patch("cognee.modules.observability.get_observe.get_base_config") as get_config:
        get_config.return_value = type("Config", (), {"monitoring_tool": Observer.LLMLITE})()

        with pytest.raises(UnsupportedObserverError, match="Unsupported observer"):
            get_observe()


def test_get_observe_langsmith_returns_callable():
    """Observer.LANGSMITH returns a callable decorator without raising."""
    mock_traceable = MagicMock(return_value=lambda func: func)

    with patch("cognee.modules.observability.get_observe.get_base_config") as get_config, \
         patch.dict("sys.modules", {"langsmith": MagicMock(traceable=mock_traceable)}):
        get_config.return_value = type("Config", (), {"monitoring_tool": Observer.LANGSMITH})()

        observe = get_observe()
        assert callable(observe)


def test_langsmith_observe_maps_as_type_to_run_type():
    """as_type values are correctly mapped to LangSmith run_type."""
    captured = {}

    def fake_traceable(**kwargs):
        captured.update(kwargs)
        return lambda func: func

    with patch("cognee.modules.observability.get_observe.get_base_config") as get_config, \
         patch.dict("sys.modules", {"langsmith": MagicMock(traceable=fake_traceable)}):
        get_config.return_value = type("Config", (), {"monitoring_tool": Observer.LANGSMITH})()

        observe = get_observe()

        def dummy(): pass

        observe(as_type="generation")(dummy)
        assert captured.get("run_type") == "llm"

        captured.clear()
        observe(as_type="retrieval")(dummy)
        assert captured.get("run_type") == "retrieval"

        captured.clear()
        observe(as_type="unknown")(dummy)
        assert captured.get("run_type") == "chain"


def test_langsmith_observe_forwards_name():
    """name kwarg is forwarded to traceable."""
    captured = {}

    def fake_traceable(**kwargs):
        captured.update(kwargs)
        return lambda func: func

    with patch("cognee.modules.observability.get_observe.get_base_config") as get_config, \
         patch.dict("sys.modules", {"langsmith": MagicMock(traceable=fake_traceable)}):
        get_config.return_value = type("Config", (), {"monitoring_tool": Observer.LANGSMITH})()

        observe = get_observe()

        def dummy(): pass
        observe(name="cognee.recall")(dummy)
        assert captured.get("name") == "cognee.recall"
        assert captured.get("run_type") == "chain"


def test_langsmith_observe_discards_unknown_kwargs():
    """Unknown kwargs are not forwarded to traceable."""
    captured = {}

    def fake_traceable(**kwargs):
        captured.update(kwargs)
        return lambda func: func

    with patch("cognee.modules.observability.get_observe.get_base_config") as get_config, \
         patch.dict("sys.modules", {"langsmith": MagicMock(traceable=fake_traceable)}):
        get_config.return_value = type("Config", (), {"monitoring_tool": Observer.LANGSMITH})()

        observe = get_observe()

        def dummy(): pass
        observe(as_type="generation", unknown_key="value")(dummy)
        assert "unknown_key" not in captured
        assert "as_type" not in captured
