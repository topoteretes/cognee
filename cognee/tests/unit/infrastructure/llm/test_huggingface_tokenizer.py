"""HuggingFaceTokenizer loads a repo's fast tokenizer without transformers (SDK-810).

transformers decides at import whether PyTorch is present, so importing it before
the GLiNER auto-installer adds torch breaks that install. The adapter therefore
counts with the ``tokenizers`` library and reaches for ``AutoTokenizer`` only when
``tokenizer.json`` cannot be loaded and transformers is already installed.
"""

import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.tokenizer.HuggingFace import adapter
from cognee.infrastructure.llm.tokenizer.HuggingFace.adapter import HuggingFaceTokenizer

_TOKENS = ["hel", "##lo", "world"]


class _FastTokenizer:
    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "counts must not include CLS/SEP"
        return SimpleNamespace(tokens=_TOKENS)


def _no_transformers():
    """Make ``import transformers`` fail, so a stray import shows up as an error."""
    return patch.dict(sys.modules, {"transformers": None, "transformers.utils": None})


def test_counts_with_tokenizers_and_never_imports_transformers():
    with (
        _no_transformers(),
        patch("huggingface_hub.hf_hub_download", return_value="/cache/tokenizer.json") as dl,
        patch("tokenizers.Tokenizer.from_file", return_value=_FastTokenizer()) as load,
    ):
        tokenizer = HuggingFaceTokenizer(model="BAAI/bge-small-en-v1.5")
        assert tokenizer.count_tokens("hello world") == 3
        assert tokenizer.extract_tokens("hello world") == _TOKENS
    dl.assert_called_once_with("BAAI/bge-small-en-v1.5", "tokenizer.json")
    load.assert_called_once_with("/cache/tokenizer.json")


def test_falls_back_to_auto_tokenizer_when_transformers_is_installed():
    fake_auto = SimpleNamespace(
        from_pretrained=lambda model: SimpleNamespace(tokenize=lambda text: _TOKENS)
    )
    with (
        patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=fake_auto)}),
        patch("huggingface_hub.hf_hub_download", side_effect=OSError("no tokenizer.json")),
        patch.object(adapter.importlib.util, "find_spec", return_value=object()),
    ):
        tokenizer = HuggingFaceTokenizer(model="org/slow-tokenizer-only")
        assert tokenizer.count_tokens("hello world") == 3


def test_raises_naming_both_the_missing_file_and_transformers_when_neither_helps():
    with (
        _no_transformers(),
        patch("huggingface_hub.hf_hub_download", side_effect=OSError("no tokenizer.json")),
        patch.object(adapter.importlib.util, "find_spec", return_value=None) as spec,
        pytest.raises(ImportError) as raised,
    ):
        HuggingFaceTokenizer(model="org/slow-tokenizer-only")
    spec.assert_called_once_with("transformers")
    message = str(raised.value)
    assert "tokenizer.json" in message and "org/slow-tokenizer-only" in message
    assert "no tokenizer.json" in message  # the underlying cause
    assert "transformers" in message and "not installed" in message


def test_find_spec_is_the_real_one():
    # The tests above patch importlib.util.find_spec on the adapter's module; make
    # sure that is the name the adapter actually calls.
    assert adapter.importlib.util.find_spec is importlib.util.find_spec
