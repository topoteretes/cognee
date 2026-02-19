"""Tests for the observability module — specifically get_observe().

Verifies that:
- get_observe() returns a valid decorator for every *supported* Observer value.
- get_observe() raises InvalidObserverError (not returns None) for unsupported
  observers (LLMLITE, LANGSMITH), preventing the silent TypeError that
  previously crashed all LLM adapters.
"""

import asyncio
import pytest
from unittest.mock import patch

from cognee.exceptions import InvalidObserverError
from cognee.modules.observability.observers import Observer
from cognee.modules.observability.get_observe import get_observe


class TestGetObserve:
    """Tests for get_observe() function."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_valid_decorator(self, observe):
        """Verify that *observe* works as both a direct and a parameterised decorator."""
        assert observe is not None, "get_observe() returned None"
        assert callable(observe), f"get_observe() returned non-callable: {type(observe)}"

        # Parameterised usage: @observe(as_type="generation")
        decorator = observe(as_type="generation")
        assert callable(decorator), "Parameterised observe() did not return a callable decorator"

        @decorator
        def parameterised_func():
            return "parameterised"

        assert parameterised_func() == "parameterised"

        # Direct usage: @observe
        @observe
        def direct_func():
            return "direct"

        assert direct_func() == "direct"

    # ------------------------------------------------------------------
    # Supported observers
    # ------------------------------------------------------------------

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_get_observe_returns_noop_for_none(self, mock_config):
        """Observer.NONE should return a working no-op decorator."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()
        self._assert_valid_decorator(observe)

    # ------------------------------------------------------------------
    # Unsupported observers — must raise InvalidObserverError, NOT return None
    # ------------------------------------------------------------------

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_get_observe_raises_for_llmlite(self, mock_config):
        """Observer.LLMLITE must raise InvalidObserverError.

        Regression test: before the fix, LLMLITE caused get_observe() to
        return None, which then crashed all LLM adapters with a TypeError
        when the None was called as a decorator.
        """
        mock_config.return_value.monitoring_tool = Observer.LLMLITE
        with pytest.raises(InvalidObserverError) as exc_info:
            get_observe()

        error = exc_info.value
        assert error.status_code == 400
        assert "LLMLITE" in error.message
        assert error.name == "InvalidObserverError"

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_get_observe_raises_for_langsmith(self, mock_config):
        """Observer.LANGSMITH must raise InvalidObserverError.

        Regression test: before the fix, LANGSMITH caused get_observe() to
        return None, which then crashed all LLM adapters with a TypeError
        when the None was called as a decorator.
        """
        mock_config.return_value.monitoring_tool = Observer.LANGSMITH
        with pytest.raises(InvalidObserverError) as exc_info:
            get_observe()

        error = exc_info.value
        assert error.status_code == 400
        assert "LANGSMITH" in error.message
        assert error.name == "InvalidObserverError"

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_invalid_observer_error_is_cognee_api_error(self, mock_config):
        """InvalidObserverError must be a CogneeApiError so FastAPI handles it correctly."""
        from cognee.exceptions import CogneeApiError

        mock_config.return_value.monitoring_tool = Observer.LLMLITE
        with pytest.raises(CogneeApiError):
            get_observe()

    # ------------------------------------------------------------------
    # No-op decorator contract
    # ------------------------------------------------------------------

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_noop_decorator_preserves_function_name(self, mock_config):
        """The no-op decorator must not alter the decorated function's __name__."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()

        @observe
        def my_function():
            """My docstring."""

        assert my_function.__name__ == "my_function"

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_noop_parameterised_decorator_preserves_function_name(self, mock_config):
        """The parameterised no-op decorator must not alter the decorated function's __name__."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()

        @observe(as_type="generation")
        def my_function():
            """My docstring."""

        assert my_function.__name__ == "my_function"

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_noop_decorator_passes_args_through(self, mock_config):
        """Decorated functions must receive their arguments unchanged."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()

        @observe(as_type="generation")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_noop_decorator_works_with_async(self, mock_config):
        """The no-op decorator must not break async functions."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()

        @observe(as_type="generation")
        async def async_func():
            return "async_result"

        result = asyncio.run(async_func())
        assert result == "async_result"