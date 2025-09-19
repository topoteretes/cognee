from enum import Enum


class Observer(str, Enum):
    """Monitoring tools"""

    NONE = "none"
    LANGFUSE = "langfuse"
    LLMLITE = "llmlite"
    LANGSMITH = "langsmith"
