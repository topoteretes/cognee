"""The two ``transformers`` names gliner2's inference path needs, without transformers.

``AutoTokenizer`` loads the checkpoint's ``tokenizer.json`` with the
``tokenizers`` library (already a cognee dependency through fastembed) and
exposes the three methods gliner2's ``SchemaTransformer`` calls. The fast
Hugging Face tokenizer runs the same ``tokenizers`` backend, so tokens and ids
are identical (checked by the parity test). ``PretrainedConfig`` is the
attribute-bag behaviour ``ExtractorConfig`` builds on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _checkpoint_file(repo_or_dir: str, name: str) -> str:
    if os.path.isdir(repo_or_dir):
        return str(Path(repo_or_dir) / name)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_or_dir, name)


class _Tokenizer:
    def __init__(self, backend):
        self._backend = backend
        self._unk_id = backend.token_to_id("[UNK]")

    def add_special_tokens(self, mapping):
        from tokenizers import AddedToken

        tokens = [
            AddedToken(t, special=True, normalized=False)
            for t in mapping.get("additional_special_tokens", [])
        ]
        return self._backend.add_special_tokens(tokens)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            found = self._backend.token_to_id(tokens)
            return self._unk_id if found is None else found
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def tokenize(self, text):
        return self._backend.encode(text, add_special_tokens=False).tokens

    def __len__(self):
        return self._backend.get_vocab_size(with_added_tokens=True)


class AutoTokenizer:
    @staticmethod
    def from_pretrained(repo_or_dir, **_):
        from tokenizers import Tokenizer

        return _Tokenizer(Tokenizer.from_file(_checkpoint_file(str(repo_or_dir), "tokenizer.json")))


class PretrainedConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**{**dict(config_dict), **kwargs})

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls.from_dict(json.loads(Path(path).read_text()), **kwargs)

    def to_dict(self):
        return dict(vars(self))


class AutoConfig(PretrainedConfig):
    pass


class _NotOnThisPath:
    """transformers model classes gliner2 imports alongside the tokenizer."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError(f"{cls.__name__} is a transformers model; the torch-free path runs ONNX")

    @classmethod
    def from_config(cls, *args, **kwargs):
        cls.from_pretrained()


class AutoModel(_NotOnThisPath):
    pass


class PreTrainedModel(_NotOnThisPath):
    pass
