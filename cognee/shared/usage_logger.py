import asyncio
import inspect
import os
from datetime import datetime, timezone
from functools import singledispatch, wraps
from typing import Any, Callable, Optional
from uuid import UUID

from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
from cognee.shared.exceptions import UsageLoggerError
from cognee.shared.logging_utils import get_logger
from cognee import __version__ as cognee_version

logger = get_logger("usage_logger")


@singledispatch
def _sanitize_value(value: Any) -> Any:
    """Default handler for JSON serialization - converts to string."""
    try:
        str_repr = str(value)
        if str_repr.startswith("<") and str_repr.endswith(">"):
            return f"<cannot be serialized: {type(value).__name__}>"
        return str_repr
    except Exception:
        return f"<cannot be serialized: {type(value).__name__}>"


@_sanitize_value.register(type(None))
def _(value: None) -> None:
    """Handle None values - returns None as-is."""
    return None


@_sanitize_value.register(str)
@_sanitize_value.register(int)
@_sanitize_value.register(float)
@_sanitize_value.register(bool)
def _(value: str | int | float | bool) -> str | int | float | bool:
    """Handle primitive types - returns value as-is since they're JSON-serializable."""
    return value


@_sanitize_value.register(UUID)
def _(value: UUID) -> str:
    """Convert UUID to string representation."""
    return str(value)


@_sanitize_value.register(datetime)
def _(value: datetime) -> str:
    """Convert datetime to ISO format string."""
    return value.isoformat()


@_sanitize_value.register(list)
@_sanitize_value.register(tuple)
def _(value: list | tuple) -> list:
    """Recursively sanitize list or tuple elements."""
    return [_sanitize_value(v) for v in value]


@_sanitize_value.register(dict)
def _(value: dict) -> dict:
    """Recursively sanitize dictionary keys and values."""
    sanitized = {}
    for k, v in value.items():
        key_str = k if isinstance(k, str) else _sanitize_dict_key(k)
        sanitized[key_str] = _sanitize_value(v)
    return sanitized


def _sanitize_dict_key(key: Any) -> str:
    """Convert a non-string dict key to a string."""
    sanitized_key = _sanitize_value(key)
    if isinstance(sanitized_key, str):
        if sanitized_key.startswith("<cannot be serialized"):
            return f"<key:{type(key).__name__}>"
        return sanitized_key
    return str(sanitized_key)


def _get_param_names(func: Callable) -> list[str]:
    """Get parameter names from function signature."""
    try:
        return list(inspect.signature(func).parameters.keys())
    except Exception:
        return []


def _get_param_defaults(func: Callable) -> dict[str, Any]:
    """Get parameter defaults from function signature."""
    try:
        sig = inspect.signature(func)
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
        return defaults
    except Exception:
        return {}


def _extract_user_id(args: tuple, kwargs: dict, param_names: list[str]) -> Optional[str]:
    """Extract user_id from function arguments if available."""
    try:
        if "user" in kwargs and kwargs["user"] is not None:
            user = kwargs["user"]
            if hasattr(user, "id"):
                return str(user.id)

        for i, param_name in enumerate(param_names):
            if i < len(args) and param_name == "user":
                user = args[i]
                if user is not None and hasattr(user, "id"):
                    return str(user.id)
        return None
    except Exception:
        return None


def _extract_parameters(args: tuple, kwargs: dict, param_names: list[str], func: Callable) -> dict:
    """Extract function parameters - captures all parameters including defaults, sanitizes for JSON."""
    params = {}

    for key, value in kwargs.items():
        if key != "user":
            params[key] = _sanitize_value(value)

    if param_names:
        for i, param_name in enumerate(param_names):
            if i < len(args) and param_name != "user" and param_name not in kwargs:
                params[param_name] = _sanitize_value(args[i])
    else:
        for i, arg_value in enumerate(args):
            params[f"arg_{i}"] = _sanitize_value(arg_value)

    if param_names:
        defaults = _get_param_defaults(func)
        for param_name in param_names:
            if param_name != "user" and param_name not in params and param_name in defaults:
                params[param_name] = _sanitize_value(defaults[param_name])

    return params


async def _log_usage_async(
    function_name: str,
    log_type: str,
    user_id: Optional[str],
    parameters: dict,
    result: Any,
    success: bool,
    error: Optional[str],
    duration_ms: float,
    start_time: datetime,
    end_time: datetime,
):
    """Asynchronously log function usage to Redis.

    Args:
        function_name: Name of the function being logged.
        log_type: Type of log entry (e.g., "api_endpoint", "mcp_tool", "function").
        user_id: User identifier, or None to use "unknown".
        parameters: Dictionary of function parameters (sanitized).
        result: Function return value (will be sanitized).
        success: Whether the function executed successfully.
        error: Error message if function failed, None otherwise.
        duration_ms: Execution duration in milliseconds.
        start_time: Function start timestamp.
        end_time: Function end timestamp.

    Note:
        This function silently handles errors to avoid disrupting the original
        function execution. Logs are written to Redis with TTL from config.
    """
    try:
        logger.debug(f"Starting to log usage for {function_name} at {start_time.isoformat()}")
        config = get_cache_config()
        if not config.usage_logging:
            logger.debug("Usage logging disabled, skipping log")
            return

        logger.debug(f"Getting cache engine for {function_name}")
        cache_engine = get_cache_engine()
        if cache_engine is None:
            logger.warning(
                f"Cache engine not available for usage logging (function: {function_name})"
            )
            return

        logger.debug(f"Cache engine obtained for {function_name}")

        if user_id is None:
            user_id = "unknown"
            logger.debug(f"No user_id provided, using 'unknown' for {function_name}")

        log_entry = {
            "timestamp": start_time.isoformat(),
            "type": log_type,
            "function_name": function_name,
            "user_id": user_id,
            "parameters": parameters,
            "result": _sanitize_value(result),
            "success": success,
            "error": error,
            "duration_ms": round(duration_ms, 2),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metadata": {
                "cognee_version": cognee_version,
                "environment": os.getenv("ENV", "prod"),
            },
        }

        logger.debug(f"Calling log_usage for {function_name}, user_id={user_id}")
        await cache_engine.log_usage(
            user_id=user_id,
            log_entry=log_entry,
            ttl=config.usage_logging_ttl,
        )
        logger.info(f"Successfully logged usage for {function_name} (user_id={user_id})")
    except Exception as e:
        logger.error(f"Failed to log usage for {function_name}: {str(e)}", exc_info=True)


def log_usage(function_name: Optional[str] = None, log_type: str = "function"):
    """
    Decorator to log function usage to Redis.

    This decorator is completely transparent - it doesn't change function behavior.
    It logs function name, parameters, result, timing, and user (if available).

    Args:
        function_name: Optional name for the function (defaults to func.__name__)
        log_type: Type of log entry (e.g., "api_endpoint", "mcp_tool")

    Usage:
        @log_usage(function_name="MCP my_mcp_tool", log_type="mcp_tool")
        async def my_mcp_tool(...):
            # mcp code

        @log_usage(function_name="POST API /v1/add", log_type="api_endpoint")
        async def add(...):
            # endpoint code
    """

    def decorator(func: Callable) -> Callable:
        """Inner decorator that wraps the function with usage logging.

        Args:
            func: The async function to wrap with usage logging.

        Returns:
            Callable: The wrapped function with usage logging enabled.

        Raises:
            UsageLoggerError: If the function is not async.
        """
        if not inspect.iscoroutinefunction(func):
            raise UsageLoggerError(
                f"@log_usage requires an async function. Got {func.__name__} which is not async."
            )

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Wrapper function that executes the original function and logs usage.

            This wrapper:
            - Extracts user ID and parameters from function arguments
            - Executes the original function
            - Captures result, success status, and any errors
            - Logs usage information asynchronously without blocking

            Args:
                *args: Positional arguments passed to the original function.
                **kwargs: Keyword arguments passed to the original function.

            Returns:
                Any: The return value of the original function.

            Raises:
                Any exception raised by the original function (re-raised after logging).
            """
            config = get_cache_config()
            if not config.usage_logging:
                return await func(*args, **kwargs)

            start_time = datetime.now(timezone.utc)

            param_names = _get_param_names(func)
            user_id = _extract_user_id(args, kwargs, param_names)
            parameters = _extract_parameters(args, kwargs, param_names, func)

            result = None
            success = True
            error = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = datetime.now(timezone.utc)
                duration_ms = (end_time - start_time).total_seconds() * 1000

                try:
                    await _log_usage_async(
                        function_name=function_name or func.__name__,
                        log_type=log_type,
                        user_id=user_id,
                        parameters=parameters,
                        result=result,
                        success=success,
                        error=error,
                        duration_ms=duration_ms,
                        start_time=start_time,
                        end_time=end_time,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to log usage for {function_name or func.__name__}: {str(e)}",
                        exc_info=True,
                    )

        return async_wrapper

    return decorator
