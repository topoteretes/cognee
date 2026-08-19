import asyncio
import inspect
import os
from collections.abc import Callable
from datetime import datetime, timezone
from functools import singledispatch, wraps
from typing import Any
from uuid import UUID

from cognee import __version__ as cognee_version
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
from cognee.shared.exceptions import UsageLoggerError
from cognee.shared.logging_utils import get_logger

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


def _get_callable_name(func: Callable) -> str:
    """Return a stable display name for functions and callable objects."""
    name = getattr(func, "__name__", None)
    if isinstance(name, str):
        return name
    return func.__class__.__name__


def _extract_user_id(args: tuple, kwargs: dict, param_names: list[str]) -> str | None:
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
    user_id: str | None,
    parameters: dict,
    result: Any,
    success: bool,
    error: str | None,
    duration_ms: float,
    start_time: datetime,
    end_time: datetime,
) -> None:
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


def _is_streaming_response(result: Any) -> bool:
    """A response whose body is produced after the handler returns.

    Duck-typed rather than importing starlette here: this module is shared and
    has no other web-framework dependency.
    """
    return hasattr(result, "body_iterator")


def _wrap_streaming_result(result: Any, emit, function_name: str):
    """Defer usage logging to the end of a streaming body.

    Returns the response with a wrapped iterator, or ``None`` if it is not a
    streaming response. The stream's own duration and outcome are what get
    logged — a client disconnect included, since that cancels the iterator.
    """
    if not _is_streaming_response(result):
        return None

    inner = result.body_iterator

    async def _logged():
        success = True
        error = None
        try:
            async for chunk in inner:
                yield chunk
        except BaseException as streaming_error:  # noqa: BLE001 - logged, then re-raised
            success = False
            error = str(streaming_error) or type(streaming_error).__name__
            raise
        finally:
            # Closing the wrapped iterator explicitly. A disconnect that lands
            # while the consumer is suspended in send() never reaches `inner`,
            # so its own cleanup — which is where a streaming endpoint releases
            # per-request resources — would never run.
            aclose = getattr(inner, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except BaseException:  # noqa: BLE001 - cleanup must not mask the outcome
                    logger.debug("Failed to close streaming body", exc_info=True)
            # Shielded because the common ending is a client disconnect, which
            # cancels this scope: an unshielded await would be cancelled at its
            # first suspension point and the record would be lost precisely for
            # the requests most worth recording. BaseException, not Exception,
            # for the same reason — CancelledError is not an Exception.
            try:
                await asyncio.shield(asyncio.ensure_future(emit(None, success, error)))
            except BaseException as log_error:  # noqa: BLE001
                logger.error(
                    f"Failed to log usage for {function_name}: {str(log_error)}",
                    exc_info=True,
                )

    result.body_iterator = _logged()
    return result


def log_usage(function_name: str | None = None, log_type: str = "function"):
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
                f"@log_usage requires an async function. Got {_get_callable_name(func)} "
                "which is not async."
            )

        resolved_function_name = function_name or _get_callable_name(func)

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

            def _emit(logged_result, logged_success, logged_error):
                end_time = datetime.now(timezone.utc)
                return _log_usage_async(
                    function_name=resolved_function_name,
                    log_type=log_type,
                    user_id=user_id,
                    parameters=parameters,
                    result=logged_result,
                    success=logged_success,
                    error=logged_error,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    start_time=start_time,
                    end_time=end_time,
                )

            deferred_to_stream = False

            try:
                result = await func(*args, **kwargs)
                streamed = _wrap_streaming_result(result, _emit, resolved_function_name)
                if streamed is not None:
                    # A streaming response returns before the work happens, so
                    # logging here would record every stream as an instant
                    # success returning an unserializable object — and a failure
                    # mid-stream as success. Logging is deferred to the end of
                    # the body instead, where the real duration and outcome are.
                    deferred_to_stream = True
                    return streamed
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                # Skipped when the body iterator took over the logging; a bare
                # return here would discard an in-flight exception.
                if not deferred_to_stream:
                    try:
                        await _emit(result, success, error)
                    except Exception as e:
                        logger.error(
                            f"Failed to log usage for {resolved_function_name}: {str(e)}",
                            exc_info=True,
                        )

        return async_wrapper

    return decorator
