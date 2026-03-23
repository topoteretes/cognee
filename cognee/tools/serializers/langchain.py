"""Serialize Cognee tools to LangChain StructuredTool format."""

from ..definitions import TOOLS


def for_langchain() -> list:
    """Convert Cognee memory tools to LangChain StructuredTool objects.

    Requires ``langchain-core`` to be installed.

    Returns
    -------
    list
        A list of ``StructuredTool`` instances ready to pass to a LangChain agent.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        raise ImportError(
            "langchain-core is required for for_langchain(). "
            "Install it with: pip install langchain-core"
        )

    result = []
    for fn in TOOLS:
        tool = StructuredTool.from_function(
            coroutine=fn,
            name=fn.__name__,
            description=fn.__doc__.split("\n\n")[0] if fn.__doc__ else "",
        )
        result.append(tool)
    return result
