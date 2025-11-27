"""Test configuration for sentiment unit tests."""

import sys
import types


dummy_tokenizers = types.ModuleType("tokenizers")
dummy_tokenizers.Tokenizer = object
dummy_tokenizers.__all__ = ["Tokenizer"]

sys.modules.setdefault("tokenizers", dummy_tokenizers)

dummy_numpy = types.ModuleType("numpy")
dummy_numpy.__all__ = []
dummy_numpy.ndarray = object
dummy_numpy.float32 = float
dummy_numpy.int32 = int

sys.modules.setdefault("numpy", dummy_numpy)
