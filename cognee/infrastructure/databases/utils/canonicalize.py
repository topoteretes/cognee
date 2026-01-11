import inspect
from typing import Mapping, Callable, Any, Dict


def canonicalize_kwargs_for_signature(
    raw_params: Mapping[str, Any],
    target_func: Callable[..., Any],
    defaults: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a canonical, ordered kwargs dict aligned with the target function's signature.
    - Merges provided raw_params over defaults
    - Keeps only parameters that the target function accepts
    - Orders keys to match the function signature to produce stable cache keys
    """
    base: Dict[str, Any] = dict(defaults or {})
    merged: Dict[str, Any] = {**base, **(raw_params or {})}

    sig = inspect.signature(target_func)
    ordered: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            if name in merged:
                ordered[name] = merged[name]
            elif param.default is not inspect._empty:
                ordered[name] = param.default
            else:
                # Ensure the key exists; None will surface missing-required issues downstream if needed
                ordered[name] = None
    return ordered
