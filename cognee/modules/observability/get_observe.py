from cognee.base_config import get_base_config
from .observers import Observer


def get_observe():
    monitoring = get_base_config().monitoring_tool

    if monitoring == Observer.LANGFUSE:
        from langfuse.decorators import observe

        return observe
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

        return no_op_decorator
