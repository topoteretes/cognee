from enum import Enum
from importlib import import_module
from typing import Optional, Type


class EvaluatorAdapter(Enum):
    """Registry of evaluation engines.

    The adapter class is resolved lazily through an import path instead of being
    imported at module load time. This keeps the optional ``deepeval`` dependency
    out of the import graph unless the DeepEval engine is actually selected, so
    importing this registry - or running the DirectLLM engine - never requires the
    ``eval`` extra to be installed.
    """

    #  (engine name, module path, class name, optional extra that provides it)
    DEEPEVAL = (
        "DeepEval",
        "cognee.eval_framework.evaluation.deep_eval_adapter",
        "DeepEvalAdapter",
        "eval",
    )
    DIRECT_LLM = (
        "DirectLLM",
        "cognee.eval_framework.evaluation.direct_llm_eval_adapter",
        "DirectLLMEvalAdapter",
        None,
    )

    def __new__(cls, adapter_name: str, module_path: str, class_name: str, extra: Optional[str]):
        obj = object.__new__(cls)
        obj._value_ = adapter_name
        obj._module_path = module_path
        obj._class_name = class_name
        obj._extra = extra
        return obj

    def load_adapter_class(self) -> Type:
        """Import and return the adapter class, raising an actionable error if the
        optional dependency backing this engine is not installed."""
        try:
            module = import_module(self._module_path)
        except ImportError as error:
            if self._extra:
                raise ImportError(
                    f"The '{self.value}' evaluation engine requires optional dependencies. "
                    f'Install them with: pip install "cognee[{self._extra}]"'
                ) from error
            raise
        return getattr(module, self._class_name)

    @property
    def adapter_class(self) -> Type:
        """Backwards-compatible accessor that resolves the adapter class lazily."""
        return self.load_adapter_class()

    def __str__(self):
        return self.value
