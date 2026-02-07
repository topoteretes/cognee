import functools

from cognee.base_config import get_base_config
from .observers import Observer


def _wrap_with_otel(inner_decorator):
    """Compose OTEL span creation around an existing decorator.

    When tracing is enabled, every decorated function is also wrapped in an
    OTEL span.  The ``as_type`` keyword (e.g. ``"generation"``) is mapped to
    a ``cognee.span.category`` attribute on the span.
    """

    def otel_observe(*dec_args, **dec_kwargs):
        # Parameterised call: @observe(as_type="generation")
        if not (len(dec_args) == 1 and callable(dec_args[0]) and not dec_kwargs):
            inner_result = inner_decorator(*dec_args, **dec_kwargs)
            category = dec_kwargs.get("as_type", "default")

            def outer(func):
                wrapped = inner_result(func) if callable(inner_result) else func

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    from cognee.modules.observability.trace_context import is_tracing_enabled

                    if not is_tracing_enabled():
                        return await wrapped(*args, **kwargs)

                    from cognee.modules.observability.tracing import (
                        get_tracer,
                        COGNEE_SPAN_CATEGORY,
                    )

                    tracer = get_tracer()
                    if tracer is None:
                        return await wrapped(*args, **kwargs)

                    with tracer.start_as_current_span(f"cognee.observe.{func.__name__}") as span:
                        span.set_attribute(COGNEE_SPAN_CATEGORY, category)
                        return await wrapped(*args, **kwargs)

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    from cognee.modules.observability.trace_context import is_tracing_enabled

                    if not is_tracing_enabled():
                        return wrapped(*args, **kwargs)

                    from cognee.modules.observability.tracing import (
                        get_tracer,
                        COGNEE_SPAN_CATEGORY,
                    )

                    tracer = get_tracer()
                    if tracer is None:
                        return wrapped(*args, **kwargs)

                    with tracer.start_as_current_span(f"cognee.observe.{func.__name__}") as span:
                        span.set_attribute(COGNEE_SPAN_CATEGORY, category)
                        return wrapped(*args, **kwargs)

                import asyncio

                if asyncio.iscoroutinefunction(func):
                    return async_wrapper
                return sync_wrapper

            return outer

        # Direct call: @observe  (no parentheses)
        func = dec_args[0]
        wrapped = inner_decorator(func) if callable(inner_decorator) else func

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            from cognee.modules.observability.trace_context import is_tracing_enabled

            if not is_tracing_enabled():
                return await wrapped(*args, **kwargs)

            from cognee.modules.observability.tracing import get_tracer

            tracer = get_tracer()
            if tracer is None:
                return await wrapped(*args, **kwargs)

            with tracer.start_as_current_span(f"cognee.observe.{func.__name__}"):
                return await wrapped(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            from cognee.modules.observability.trace_context import is_tracing_enabled

            if not is_tracing_enabled():
                return wrapped(*args, **kwargs)

            from cognee.modules.observability.tracing import get_tracer

            tracer = get_tracer()
            if tracer is None:
                return wrapped(*args, **kwargs)

            with tracer.start_as_current_span(f"cognee.observe.{func.__name__}"):
                return wrapped(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return otel_observe


def get_observe():
    monitoring = get_base_config().monitoring_tool

    if monitoring == Observer.LANGFUSE:
        from langfuse.decorators import observe

        return _wrap_with_otel(observe)
    elif monitoring == Observer.NONE:
        # Return a no-op decorator that handles keyword arguments
        def no_op_decorator(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                # Direct decoration: @observe
                return args[0]
            else:
                # Parameterized decoration: @observe(as_type="generation")
                def decorator(func):
                    return func

                return decorator

        return _wrap_with_otel(no_op_decorator)
