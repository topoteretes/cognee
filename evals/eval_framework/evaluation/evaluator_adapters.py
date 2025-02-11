from enum import Enum
from typing import Type
from evals.eval_framework.evaluation.deep_eval_adapter import DeepEvalAdapter


class EvaluatorAdapter(Enum):
    DEEPEVAL = ("DeepEval", DeepEvalAdapter)

    def __new__(cls, adapter_name: str, adapter_class: Type):
        obj = object.__new__(cls)
        obj._value_ = adapter_name
        obj.adapter_class = adapter_class
        return obj

    def __str__(self):
        return self.value
