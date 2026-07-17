import importlib
from enum import Enum


class EvaluatorAdapter(Enum):
    """Adapter class is resolved lazily on first ``.adapter_class`` access, not at module
    import time — so selecting one adapter (e.g. ``BeamEval``) never forces every other
    adapter's dependencies (e.g. ``deepeval``) to be installed.
    """

    DEEPEVAL = ("DeepEval", "cognee.eval_framework.evaluation.deep_eval_adapter", "DeepEvalAdapter")
    BEAM = ("BeamEval", "cognee.eval_framework.beam.eval.beam_eval_adapter", "BeamEvalAdapter")
    DIRECT_LLM = (
        "DirectLLM",
        "cognee.eval_framework.evaluation.direct_llm_eval_adapter",
        "DirectLLMEvalAdapter",
    )

    def __new__(cls, adapter_name: str, module_path: str, class_name: str):
        obj = object.__new__(cls)
        obj._value_ = adapter_name
        obj._module_path = module_path
        obj._class_name = class_name
        return obj

    @property
    def adapter_class(self):
        return getattr(importlib.import_module(self._module_path), self._class_name)

    def __str__(self):
        return self.value
