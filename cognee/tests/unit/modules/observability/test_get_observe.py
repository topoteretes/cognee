"""Tests for the observability module — specifically get_observe().

Verifies that get_observe() returns a valid decorator for every Observer
enum value, preventing the TypeError crash that occurred when unsupported
observers (LLMLITE, LANGSMITH) caused it to return None.
"""

import pytest
from unittest.mock import patch
from cognee.modules.observability.observers import Observer
from cognee.modules.observability.get_observe import get_observe


class TestGetObserve:
    """Tests for get_observe() function."""

    def _assert_valid_decorator(self, observe):
        """Helper: verify that `observe` works as both direct and parameterized decorator."""
        # Must not be None
        assert observe is not None, "get_observe() returned None"

        # Must be callable
        assert callable(observe), f"get_observe() returned non-callable: {type(observe)}"

        # Parameterized usage: @observe(as_type="generation")
        decorator = observe(as_type="generation")
        assert callable(decorator), "Parameterized observe() did not return a callable decorator"

        @decorator
        def parameterized_func():
            return "parameterized"

        assert parameterized_func() == "parameterized"

        # Direct usage: @observe
        @observe
        def direct_func():
            return "direct"

        assert direct_func() == "direct"

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_get_observe_returns_noop_for_none(self, mock_config):
        """Observer.NONE should return a working no-op decorator."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()
        self._assert_valid_decorator(observe)

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_get_observe_returns_noop_for_llmlite(self, mock_config):
        """Observer.LLMLITE should return a working no-op decorator (not None).

        This is the bug regression test: before the fix, LLMLITE caused
        get_observe() to return None, crashing all LLM adapters.
        """
        mock_config.return_value.monitoring_tool = Observer.LLMLITE
        observe = get_observe()
        self._assert_valid_decorator(observe)

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_get_observe_returns_noop_for_langsmith(self, mock_config):
        """Observer.LANGSMITH should return a working no-op decorator (not None).

        This is the bug regression test: before the fix, LANGSMITH caused
        get_observe() to return None, crashing all LLM adapters.
        """
        mock_config.return_value.monitoring_tool = Observer.LANGSMITH
        observe = get_observe()
        self._assert_valid_decorator(observe)

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_get_observe_never_returns_none_for_any_observer(self, mock_config):
        """Exhaustive check: get_observe() must never return None for any Observer value."""
        for observer in Observer:
            if observer == Observer.LANGFUSE:
                continue  # Skip LANGFUSE — it requires the langfuse package
            mock_config.return_value.monitoring_tool = observer
            observe = get_observe()
            assert observe is not None, (
                f"get_observe() returned None for Observer.{observer.name}"
            )
            assert callable(observe), (
                f"get_observe() returned non-callable for Observer.{observer.name}"
            )

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_noop_decorator_preserves_function_name(self, mock_config):
        """The no-op decorator should not alter the decorated function's identity."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()

        @observe
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_noop_parameterized_decorator_preserves_function_name(self, mock_config):
        """The parameterized no-op decorator should not alter the decorated function's identity."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()

        @observe(as_type="generation")
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_noop_decorator_passes_args_through(self, mock_config):
        """Decorated functions should receive their arguments unchanged."""
        mock_config.return_value.monitoring_tool = Observer.NONE
        observe = get_observe()

        @observe(as_type="generation")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    @patch("cognee.modules.observability.get_observe.get_base_config")
    def test_noop_decorator_works_with_async(self, mock_config):
        """The no-op decorator should not break async functions."""
        import asyncio

        mock_config.return_value.monitoring_tool = Observer.LLMLITE
        observe = get_observe()

        @observe(as_type="generation")
        async def async_func():
            return "async_result"

        result = asyncio.run(async_func())
        assert result == "async_result"
