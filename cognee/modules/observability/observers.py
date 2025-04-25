from enum import Enum


class Observer(str, Enum):
    """Monitoring tools"""

    LANGFUSE = "langfuse"
    LLMLITE = "llmlite"
    LANGSMITH = "langsmith"
