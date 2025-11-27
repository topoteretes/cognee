"""Global test configuration to avoid optional binary dependencies."""

import sys
import types


def _ensure_module(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


dummy_tokenizers = types.ModuleType("tokenizers")
dummy_tokenizers.Tokenizer = object
dummy_tokenizers.__all__ = ["Tokenizer"]
_ensure_module("tokenizers", dummy_tokenizers)


class _DummyArray(list):
    """Minimal stand-in for numpy.ndarray used in tests."""

    def astype(self, _dtype):
        return self


def _array(data, dtype=None):  # noqa: D401 - simple shim
    if isinstance(data, _DummyArray):
        return data
    converted = _DummyArray(data)
    if dtype in (float, "float32"):
        return _DummyArray(float(x) for x in converted)
    return converted


dummy_numpy = types.ModuleType("numpy")
dummy_numpy.ndarray = _DummyArray
dummy_numpy.array = _array
dummy_numpy.float32 = float
dummy_numpy.int32 = int
dummy_numpy.zeros = lambda shape, dtype=float: _DummyArray([dtype() for _ in range(shape[0])])  # type: ignore
dummy_numpy.__all__ = [
    "ndarray",
    "array",
    "float32",
    "int32",
    "zeros",
]
_ensure_module("numpy", dummy_numpy)
