"""Flatten nested dataclass/tuple/dict/tensor structures to ``(spec, tensors)`` and back.

The boundary head returns dataclasses (``ExtractorOutput`` holding a
``CandidateTensorBatch``); an ONNX graph returns a flat tuple. The spec, saved
next to the graph, rebuilds the exact structure from that tuple.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import torch


def flatten(obj: Any) -> tuple[dict, list[torch.Tensor]]:
    tensors: list[torch.Tensor] = []

    def go(o):
        if isinstance(o, torch.Tensor):
            tensors.append(o)
            return {"t": len(tensors) - 1}
        if o is None or isinstance(o, (bool, int, float, str)):
            return {"c": o}
        if isinstance(o, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            # Shape-derived scalar under torch.export (ExtractorOutput.batch_size):
            # not a graph output; the runtime recomputes it from tensor shapes.
            return {"sym": True}
        if dataclasses.is_dataclass(o):
            cls = type(o)
            return {
                "dc": f"{cls.__module__}:{cls.__qualname__}",
                "f": {f.name: go(getattr(o, f.name)) for f in dataclasses.fields(o)},
            }
        if isinstance(o, (list, tuple)):
            return {"l" if isinstance(o, list) else "tu": [go(x) for x in o]}
        if isinstance(o, dict):
            return {"d": {str(k): go(v) for k, v in o.items()}}
        raise TypeError(f"cannot flatten {type(o).__name__}")

    return go(obj), tensors


def _resolve(path: str):
    module, qualname = path.split(":")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def unflatten(spec: dict, tensors: list, symbols: dict | None = None) -> Any:
    """Rebuild the structure; ``symbols`` fills fields that were symbolic at export."""
    symbols = symbols or {}

    def go(s, name=None):
        if "t" in s:
            return tensors[s["t"]]
        if "c" in s:
            return s["c"]
        if "sym" in s:
            return symbols[name]
        if "dc" in s:
            return _resolve(s["dc"])(**{k: go(v, k) for k, v in s["f"].items()})
        if "l" in s:
            return [go(x) for x in s["l"]]
        if "tu" in s:
            return tuple(go(x) for x in s["tu"])
        if "d" in s:
            return {k: go(v, k) for k, v in s["d"].items()}
        raise ValueError(f"bad spec node: {s}")

    return go(spec)
