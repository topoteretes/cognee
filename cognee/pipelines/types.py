"""
Type annotations for the simplified pipeline API.

- Drop: Sentinel value to filter items out of the pipeline
"""


class _Drop:
    """Sentinel value: return Drop from a step to filter an item out of the pipeline.

    Example:
        async def filter_short(text: str) -> str:
            if len(text) < 10:
                return Drop  # This item is removed from the pipeline
            return text
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Drop"

    def __bool__(self):
        return False


Drop = _Drop()
