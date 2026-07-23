import functools

from cognee.base_config import get_base_config
from .observers import Observer
from .exceptions import UnsupportedObserverError


# Cap span input/output like the DB adapters cap query text (redact_secrets(query[:500])).
_MAX_OBSERVED_CHARS = 8000


def _set_generation_attributes(span, adapter, func, args, kwargs) -> None:
    """Emit OTel-GenAI + Langfuse attributes on a generation span so any OTLP backend
    (Langfuse, Dash0, Datadog, ...) renders the LLM call as a generation with model
    and input.

    ``langfuse.observation.type`` is set unconditionally so the span is still
    classified as a generation when the model name is unavailable (e.g. llama.cpp
    local mode leaves ``model=None``). ``gen_ai.request.model`` is the *requested*
    model; provider names follow the lowercase GenAI convention.
    """
    from cognee.modules.observability.tracing import (
        GEN_AI_REQUEST_MODEL,
        GEN_AI_SYSTEM,
        LANGFUSE_OBSERVATION_TYPE,
        LANGFUSE_OBSERVATION_INPUT,
        redact_secrets,
    )

    span.set_attribute(LANGFUSE_OBSERVATION_TYPE, "generation")

    model = getattr(adapter, "model", None)
    if model:
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)

    provider = getattr(adapter, "name", None)
    if provider:
        span.set_attribute(GEN_AI_SYSTEM, provider.lower())

    payload = _generation_input_payload(func, args, kwargs)
    if payload:
        span.set_attribute(
            LANGFUSE_OBSERVATION_INPUT, redact_secrets(payload[:_MAX_OBSERVED_CHARS])
        )


def _generation_input_payload(func, args, kwargs):
    """Bind the LLM-adapter call and record its string prompt arguments as a JSON string,
    keyed by parameter name. Works for positional and keyword calls, and is name-agnostic
    so it keeps capturing the prompt if the adapter parameters are renamed. Skips
    ``self``/``response_model`` and non-string args (the response-model type, numeric
    options, the ``**kwargs`` dict). Returns None if the call can't be interpreted."""
    import json
    import inspect

    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        payload = {
            name: value
            for name, value in bound.arguments.items()
            if name not in ("self", "cls", "response_model") and isinstance(value, str) and value
        }
        return json.dumps(payload, default=str) if payload else None
    except Exception:
        return None


def _set_generation_output(span, result) -> None:
    """Record the LLM response as the Langfuse output (best effort; never raises)."""
    import json

    from cognee.modules.observability.tracing import LANGFUSE_OBSERVATION_OUTPUT, redact_secrets

    try:
        if hasattr(result, "model_dump_json"):  # a pydantic structured output
            output = result.model_dump_json()
        elif isinstance(result, str):
            output = result
        else:
            output = json.dumps(result, default=str)
        span.set_attribute(
            LANGFUSE_OBSERVATION_OUTPUT, redact_secrets(output[:_MAX_OBSERVED_CHARS])
        )
    except Exception:
        pass


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

                    from opentelemetry.trace import SpanKind
                    from cognee.modules.observability.tracing import (
                        get_tracer,
                        COGNEE_SPAN_CATEGORY,
                    )

                    tracer = get_tracer()
                    if tracer is None:
                        return await wrapped(*args, **kwargs)

                    kind = SpanKind.CLIENT if category == "generation" else SpanKind.INTERNAL
                    with tracer.start_as_current_span(
                        f"cognee.observe.{func.__name__}", kind=kind
                    ) as span:
                        span.set_attribute(COGNEE_SPAN_CATEGORY, category)
                        is_generation = category == "generation" and bool(args)
                        if is_generation:
                            _set_generation_attributes(span, args[0], func, args, kwargs)
                        result = await wrapped(*args, **kwargs)
                        if is_generation:
                            _set_generation_output(span, result)
                        return result

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    from cognee.modules.observability.trace_context import is_tracing_enabled

                    if not is_tracing_enabled():
                        return wrapped(*args, **kwargs)

                    from opentelemetry.trace import SpanKind
                    from cognee.modules.observability.tracing import (
                        get_tracer,
                        COGNEE_SPAN_CATEGORY,
                    )

                    tracer = get_tracer()
                    if tracer is None:
                        return wrapped(*args, **kwargs)

                    kind = SpanKind.CLIENT if category == "generation" else SpanKind.INTERNAL
                    with tracer.start_as_current_span(
                        f"cognee.observe.{func.__name__}", kind=kind
                    ) as span:
                        span.set_attribute(COGNEE_SPAN_CATEGORY, category)
                        is_generation = category == "generation" and bool(args)
                        if is_generation:
                            _set_generation_attributes(span, args[0], func, args, kwargs)
                        result = wrapped(*args, **kwargs)
                        if is_generation:
                            _set_generation_output(span, result)
                        return result

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

    if monitoring == Observer.NONE:
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

    else:
        observer_str = getattr(monitoring, "value", str(monitoring))
        raise UnsupportedObserverError(observer_str)
