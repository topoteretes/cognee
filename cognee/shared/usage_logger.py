import asyncio
import inspect
import os
from datetime import datetime, timezone
from functools import singledispatch, wraps
from typing import Any, Callable, Optional
from uuid import UUID

from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
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
    return None


@_sanitize_value.register(str)
@_sanitize_value.register(int)
@_sanitize_value.register(float)
@_sanitize_value.register(bool)
def _(value: str | int | float | bool) -> str | int | float | bool:
    return value


@_sanitize_value.register(UUID)
def _(value: UUID) -> str:
    return str(value)


@_sanitize_value.register(datetime)
def _(value: datetime) -> str:
    return value.isoformat()


@_sanitize_value.register(list)
@_sanitize_value.register(tuple)
def _(value: list | tuple) -> list:
    return [_sanitize_value(v) for v in value]


@_sanitize_value.register(dict)
def _(value: dict) -> dict:
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
    """Asynchronously log function usage to Redis."""
    try:
        logger.debug(f"Starting to log usage for {function_name} at {start_time.isoformat()}")
        config = get_cache_config()
        if not config.usage_logging:
            logger.debug("Usage logging disabled, skipping log")
            return

        logger.debug(f"Getting cache engine for {function_name}")
        cache_engine = get_cache_engine(lock_key=None, log_key="usage_logging")
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
        log_type: Type of log entry (e.g., "api_endpoint", "mcp_tool", "function")

    Usage:
        @log_usage()
        async def my_function(...):
            # function code

        @log_usage(function_name="POST /v1/add", log_type="api_endpoint")
        async def add(...):
            # endpoint code
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
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


if __name__ == "__main__":
    # Example 1: Simple function with decorator
    @log_usage(function_name="example_function", log_type="example")
    async def example_function(param1: str, param2: int, user=None):
        """Example function to demonstrate usage logging."""
        await asyncio.sleep(0.1)  # Simulate some work
        return {(1, 2): "ok"}

    # Example 2: Function with user parameter
    class MockUser:
        def __init__(self, user_id: str):
            self.id = user_id

    @log_usage(function_name="example_with_user", log_type="example")
    async def example_with_user(data: str, user: MockUser, wrong_param=datetime.utcnow().isoformat()):
        """Example function with user parameter."""
        await asyncio.sleep(0.05)
        return float("nan")


    @log_usage(function_name="returns_cycle", log_type="function")
    async def returns_cycle():
        a = []
        a.append(a)
        return a

    async def run_example():
        """Run example demonstrations."""
        print("Usage Logger Example")
        print("=" * 50)

        # Example 1: Simple function
        print("\n1. Running example function:")
        result1 = await example_function("example_data", 42)
        print(f"   Result: {result1}")
        await asyncio.sleep(0.2)  # Wait for async logging to complete

        # Example 2: Function with user
        print("\n2. Running example function with user:")
        mock_user = MockUser("example-user-123")
        result2 = await example_with_user("sample_data", user=mock_user, wrong_param=datetime.utcnow().isoformat())
        result3 = await example_with_user("sample_data", user=mock_user)

        print(f"   Result: {result2}")
        await asyncio.sleep(0.2)  # Wait for async logging to complete

        await returns_cycle()

        # Example 3: Retrieve logs (if cache engine is available)
        print("\n3. Retrieving usage logs:")
        try:
            config = get_cache_config()
            if config.usage_logging:
                cache_engine = get_cache_engine(lock_key="usage_logging")
                if cache_engine:
                    # Get logs for the user
                    user_id = str(mock_user.id)
                    logs = await cache_engine.get_usage_logs(user_id, limit=10)
                    print(f"   Found {len(logs)} log entries for user {user_id}")
                    if logs:
                        print(
                            f"   Latest log: {logs[0]['function_name']} at {logs[0]['timestamp']}"
                        )
                else:
                    print("   Cache engine not available")
            else:
                print("   Usage logging is disabled (set USAGE_LOGGING=true)")
        except Exception as e:
            print(f"   Error retrieving logs: {str(e)}")

        print("\n" + "=" * 50)
        print("Example completed!")
        print("\nNote: Make sure to set these environment variables:")
        print("  - USAGE_LOGGING=true")
        print("  - CACHING=true (or ensure cache backend is configured)")
        print("  - CACHE_BACKEND=redis (or fs)")
        print("  - CACHE_HOST=localhost")
        print("  - CACHE_PORT=6379")



    # Run the example
    asyncio.run(run_example())
