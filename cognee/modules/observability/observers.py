from enum import Enum


class Observer(str, Enum):
    """Monitoring tools"""

    NONE = "none"
    LLMLITE = "llmlite"
    LANGSMITH = "langsmith"
