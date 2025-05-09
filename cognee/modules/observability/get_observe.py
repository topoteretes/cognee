from cognee.base_config import get_base_config
from .observers import Observer


def get_observe():
    monitoring = get_base_config().monitoring_tool

    if monitoring == Observer.LANGFUSE:
        from langfuse.decorators import observe

        return observe
