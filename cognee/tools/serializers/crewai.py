"""Serialize Cognee tools to CrewAI BaseTool format."""

import asyncio
import inspect
from typing import Type, get_type_hints

from pydantic import BaseModel, Field

from ..definitions import TOOLS


def _build_args_schema(fn) -> Type[BaseModel]:
    """Dynamically build a Pydantic model from a function's signature."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields = {}

    for name, param in sig.parameters.items():
        hint = hints.get(name, str)
        if param.default is not inspect.Parameter.empty:
            fields[name] = (hint, Field(default=param.default, description=name))
        else:
            fields[name] = (hint, Field(..., description=name))

    model_name = f"{fn.__name__.title().replace('_', '')}Input"
    return type(
        model_name,
        (BaseModel,),
        {
            "__annotations__": {k: v[0] for k, v in fields.items()},
            **{k: v[1] for k, v in fields.items()},
        },
    )


def for_crewai() -> list:
    """Convert Cognee memory tools to CrewAI BaseTool objects.

    Requires ``crewai`` to be installed.

    Returns
    -------
    list
        A list of ``BaseTool`` instances ready to pass to a CrewAI agent.
    """
    try:
        from crewai.tools import BaseTool
    except ImportError:
        raise ImportError(
            "crewai is required for for_crewai(). Install it with: pip install crewai"
        )

    result = []
    for fn in TOOLS:
        args_schema = _build_args_schema(fn)
        async_fn = fn

        class CogneeTool(BaseTool):
            name: str = fn.__name__
            description: str = fn.__doc__.split("\n\n")[0] if fn.__doc__ else ""
            args_schema: Type[BaseModel] = args_schema
            _async_fn = async_fn

            def _run(self, **kwargs) -> str:
                return asyncio.run(self._async_fn(**kwargs))

        result.append(CogneeTool())
    return result
