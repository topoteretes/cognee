"""HuggingFaceTokenizer counts with a repo's fast tokenizer, without transformers (SDK-810).

transformers decides at import whether PyTorch is present, so importing it before
the GLiNER auto-installer adds torch breaks that install. The adapter therefore
counts with the ``tokenizers`` library, reads the model's declared input limit
from ``tokenizer_config.json`` itself, and reaches for ``AutoTokenizer`` only when
``tokenizer.json`` cannot be loaded and transformers is already installed.

The repo files come from a fake ``hf_hub_download``; the tokenizer itself is real.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from huggingface_hub.errors import EntryNotFoundError

from cognee.infrastructure.llm.tokenizer.HuggingFace import adapter
from cognee.infrastructure.llm.tokenizer.HuggingFace.adapter import HuggingFaceTokenizer

TWELVE_WORDS = " ".join(f"w{i}" for i in range(12))


@pytest.fixture
def tokenizer_json(tmp_path):
    """A real tokenizer.json: [CLS] $A [SEP] around the text (2 special tokens) and,
    like sentence-transformers/all-MiniLM-L6-v2, stored truncation and fixed padding."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    words = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", *[f"w{i}" for i in range(20)]]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.enable_truncation(max_length=4)
    tokenizer.enable_padding(length=4, pad_id=1, pad_token="[PAD]")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    assert len(Tokenizer.from_file(str(path)).encode("w1").ids) == 4, "fixture must pad"
    return str(path)


@pytest.fixture
def tokenizer_config(tmp_path):
    def write(config: dict | None):
        if config is None:
            return None
        path = tmp_path / "tokenizer_config.json"
        path.write_text(json.dumps(config))
        return str(path)

    return write


def _hub(tokenizer_json=None, tokenizer_config=None):
    """Fake hf_hub_download for the two files the adapter asks for; None = missing."""

    def download(repo, filename):
        if filename == "tokenizer.json":
            if tokenizer_json is None:
                raise OSError("no tokenizer.json")
            return tokenizer_json
        if filename == "tokenizer_config.json":
            if tokenizer_config is None:
                raise EntryNotFoundError("no tokenizer_config.json")
            return tokenizer_config
        raise AssertionError(f"unexpected file {filename}")

    return patch("huggingface_hub.hf_hub_download", side_effect=download)


def _no_transformers():
    """Make ``import transformers`` fail, so a stray import shows up as an error."""
    return patch.dict(sys.modules, {"transformers": None, "transformers.utils": None})


def test_counts_with_tokenizers_and_never_imports_transformers(tokenizer_json, tokenizer_config):
    with _no_transformers(), _hub(tokenizer_json, tokenizer_config({"model_max_length": 512})):
        tokenizer = HuggingFaceTokenizer(model="org/model")
    assert tokenizer.count_tokens("w1 w2 w3") == 3  # no [CLS]/[SEP] in the count
    assert tokenizer.extract_tokens("w1 w2") == ["w1", "w2"]


def test_counts_ignore_truncation_and_padding_stored_in_tokenizer_json(
    tokenizer_json, tokenizer_config
):
    with _no_transformers(), _hub(tokenizer_json, tokenizer_config(None)):
        tokenizer = HuggingFaceTokenizer(model="org/minilm-like")
    assert tokenizer.count_tokens("w1") == 1, "fixed padding must not inflate short texts"
    assert tokenizer.count_tokens(TWELVE_WORDS) == 12, "stored truncation must not cap long texts"


def test_model_input_limit_is_the_declared_limit_less_the_models_special_tokens(
    tokenizer_json, tokenizer_config
):
    with _no_transformers(), _hub(tokenizer_json, tokenizer_config({"model_max_length": 512})):
        assert HuggingFaceTokenizer(model="org/model").model_input_limit == 510


@pytest.mark.parametrize(
    "config",
    [None, {}, {"model_max_length": 1e30}, {"model_max_length": True}, {"model_max_length": 0}],
    ids=["no tokenizer_config.json", "no key", "float placeholder", "bool", "zero"],
)
def test_model_input_limit_is_none_when_the_repo_declares_no_usable_limit(
    tokenizer_json, tokenizer_config, config
):
    with _no_transformers(), _hub(tokenizer_json, tokenizer_config(config)):
        assert HuggingFaceTokenizer(model="org/model").model_input_limit is None


def test_falls_back_to_auto_tokenizer_when_transformers_is_installed(tokenizer_config):
    auto = SimpleNamespace(
        tokenize=lambda text: text.split(), num_special_tokens_to_add=lambda pair: 1
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda m: auto)
    )
    with (
        patch.dict(sys.modules, {"transformers": fake_transformers}),
        patch.object(adapter.importlib.util, "find_spec", return_value=object()),
        _hub(None, tokenizer_config({"model_max_length": 8192})),
    ):
        tokenizer = HuggingFaceTokenizer(model="org/slow-tokenizer-only")
    assert tokenizer.count_tokens("a b c") == 3
    assert tokenizer.model_input_limit == 8191


def test_raises_naming_both_the_missing_file_and_transformers_when_neither_helps():
    with (
        _no_transformers(),
        _hub(None, None),
        patch.object(adapter.importlib.util, "find_spec", return_value=None) as spec,
        pytest.raises(ImportError) as raised,
    ):
        HuggingFaceTokenizer(model="org/slow-tokenizer-only")
    spec.assert_called_once_with("transformers")
    message = str(raised.value)
    assert "tokenizer.json" in message and "org/slow-tokenizer-only" in message
    assert "no tokenizer.json" in message  # the underlying cause
    assert "transformers" in message and "not installed" in message
    assert raised.value.name == "transformers"  # what the resolver's hint keys on
