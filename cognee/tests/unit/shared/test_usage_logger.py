"""Unit tests for usage logger core functions."""

import pytest
from datetime import datetime, timezone
from uuid import UUID
from types import SimpleNamespace

from cognee.shared.usage_logger import (
    _sanitize_value,
    _sanitize_dict_key,
    _get_param_names,
    _get_param_defaults,
    _extract_user_id,
    _extract_parameters,
    log_usage,
)
from cognee.shared.exceptions import UsageLoggerError


class TestSanitizeValue:
    """Test _sanitize_value function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            ("string", "string"),
            (42, 42),
            (3.14, 3.14),
            (True, True),
            (False, False),
        ],
    )
    def test_basic_types(self, value, expected):
        assert _sanitize_value(value) == expected

    def test_uuid_and_datetime(self):
        """Test UUID and datetime serialization."""
        uuid_val = UUID("123e4567-e89b-12d3-a456-426614174000")
        dt = datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)

        assert _sanitize_value(uuid_val) == "123e4567-e89b-12d3-a456-426614174000"
        assert _sanitize_value(dt) == "2024-01-15T12:30:45+00:00"

    def test_collections(self):
        """Test list, tuple, and dict serialization."""
        assert _sanitize_value(
            [1, "string", UUID("123e4567-e89b-12d3-a456-426614174000"), None]
        ) == [1, "string", "123e4567-e89b-12d3-a456-426614174000", None]
        assert _sanitize_value((1, "string", True)) == [1, "string", True]
        assert _sanitize_value({"key": UUID("123e4567-e89b-12d3-a456-426614174000")}) == {
            "key": "123e4567-e89b-12d3-a456-426614174000"
        }
        assert _sanitize_value([]) == []
        assert _sanitize_value({}) == {}

    def test_nested_and_complex(self):
        """Test nested structures and non-serializable types."""
        # Nested structure
        nested = {"level1": {"level2": {"level3": [1, 2, {"nested": "value"}]}}}
        assert _sanitize_value(nested)["level1"]["level2"]["level3"][2]["nested"] == "value"

        # Non-serializable
        class CustomObject:
            def __str__(self):
                return "<CustomObject instance>"

        result = _sanitize_value(CustomObject())
        assert isinstance(result, str)
        assert "<cannot be serialized" in result or "<CustomObject" in result


class TestSanitizeDictKey:
    """Test _sanitize_dict_key function."""

    @pytest.mark.parametrize(
        "key,expected_contains",
        [
            ("simple_key", "simple_key"),
            (UUID("123e4567-e89b-12d3-a456-426614174000"), "123e4567-e89b-12d3-a456-426614174000"),
            ((1, 2, 3), ["1", "2"]),
        ],
    )
    def test_key_types(self, key, expected_contains):
        result = _sanitize_dict_key(key)
        assert isinstance(result, str)
        if isinstance(expected_contains, list):
            assert all(item in result for item in expected_contains)
        else:
            assert expected_contains in result

    def test_non_serializable_key(self):
        class BadKey:
            def __str__(self):
                return "<BadKey instance>"

        result = _sanitize_dict_key(BadKey())
        assert isinstance(result, str)
        assert "<key:" in result or "<BadKey" in result


class TestGetParamNames:
    """Test _get_param_names function."""

    @pytest.mark.parametrize(
        "func_def,expected",
        [
            (lambda a, b, c: None, ["a", "b", "c"]),
            (lambda a, b=42, c="default": None, ["a", "b", "c"]),
            (lambda a, **kwargs: None, ["a", "kwargs"]),
            (lambda *args: None, ["args"]),
        ],
    )
    def test_param_extraction(self, func_def, expected):
        assert _get_param_names(func_def) == expected

    def test_async_function(self):
        async def func(a, b):
            pass

        assert _get_param_names(func) == ["a", "b"]


class TestGetParamDefaults:
    """Test _get_param_defaults function."""

    @pytest.mark.parametrize(
        "func_def,expected",
        [
            (lambda a, b=42, c="default", d=None: None, {"b": 42, "c": "default", "d": None}),
            (lambda a, b, c: None, {}),
            (lambda a, b=10, c="test", d=None: None, {"b": 10, "c": "test", "d": None}),
        ],
    )
    def test_default_extraction(self, func_def, expected):
        assert _get_param_defaults(func_def) == expected


class TestExtractUserId:
    """Test _extract_user_id function."""

    def test_user_extraction(self):
        """Test extracting user_id from kwargs and args."""
        user1 = SimpleNamespace(id=UUID("123e4567-e89b-12d3-a456-426614174000"))
        user2 = SimpleNamespace(id="user-123")

        # From kwargs
        assert (
            _extract_user_id((), {"user": user1}, ["user", "other"])
            == "123e4567-e89b-12d3-a456-426614174000"
        )
        # From args
        assert _extract_user_id((user2, "other"), {}, ["user", "other"]) == "user-123"
        # Not present
        assert _extract_user_id(("arg1",), {}, ["param1"]) is None
        # None value
        assert _extract_user_id((None,), {}, ["user"]) is None
        # No id attribute
        assert _extract_user_id((SimpleNamespace(name="test"),), {}, ["user"]) is None


class TestExtractParameters:
    """Test _extract_parameters function."""

    def test_parameter_extraction(self):
        """Test parameter extraction with various scenarios."""

        def func1(param1, param2, user=None):
            pass

        def func2(param1, param2=42, param3="default", user=None):
            pass

        def func3():
            pass

        def func4(param1, user):
            pass

        # Kwargs only
        result = _extract_parameters(
            (), {"param1": "v1", "param2": 42}, _get_param_names(func1), func1
        )
        assert result == {"param1": "v1", "param2": 42}
        assert "user" not in result

        # Args only
        result = _extract_parameters(("v1", 42), {}, _get_param_names(func1), func1)
        assert result == {"param1": "v1", "param2": 42}

        # Mixed args/kwargs
        result = _extract_parameters(("v1",), {"param3": "v3"}, _get_param_names(func2), func2)
        assert result["param1"] == "v1" and result["param3"] == "v3"

        # Defaults included
        result = _extract_parameters(("v1",), {}, _get_param_names(func2), func2)
        assert result["param1"] == "v1" and result["param2"] == 42 and result["param3"] == "default"

        # No parameters
        assert _extract_parameters((), {}, _get_param_names(func3), func3) == {}

        # User excluded
        user = SimpleNamespace(id="user-123")
        result = _extract_parameters(("v1", user), {}, _get_param_names(func4), func4)
        assert result == {"param1": "v1"} and "user" not in result

        # Fallback when inspection fails
        class BadFunc:
            pass

        result = _extract_parameters(("arg1", "arg2"), {}, [], BadFunc())
        assert "arg_0" in result or "arg_1" in result


class TestDecoratorValidation:
    """Test decorator validation and behavior."""

    def test_decorator_validation(self):
        """Test decorator validation and metadata preservation."""
        # Sync function raises error
        with pytest.raises(UsageLoggerError, match="requires an async function"):

            @log_usage()
            def sync_func():
                pass

        # Async function accepted
        @log_usage()
        async def async_func():
            pass

        assert callable(async_func)

        # Metadata preserved
        @log_usage(function_name="test_func", log_type="test")
        async def test_func(param1: str, param2: int = 42):
            """Test docstring."""
            return param1

        assert test_func.__name__ == "test_func"
        assert "Test docstring" in test_func.__doc__
