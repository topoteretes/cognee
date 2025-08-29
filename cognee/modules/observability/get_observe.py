from cognee.base_config import get_base_config
from .observers import Observer


def get_observe():
    monitoring = get_base_config().monitoring_tool

    if monitoring == Observer.LANGFUSE:
        try:
            from langfuse.decorators import observe

            return observe
        except ImportError:
            # Return a no-op decorator if Langfuse is not available
            def noop_observe(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            return noop_observe
