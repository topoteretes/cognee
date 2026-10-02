import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..tokenizer_interface import TokenizerInterface


def _load(model: str) -> tuple[Callable[[str], list[str]], int]:
    """Return ``(tokenize, special_tokens)`` for a HuggingFace repo: ``text -> tokens``
    without special tokens, and how many special tokens the model adds around one
    text (counted text excludes them, the model's input limit includes them).

    The repo's ``tokenizer.json`` (the fast tokenizer every current embedding model
    ships) is loaded with the ``tokenizers`` library. A repo without one needs
    ``transformers.AutoTokenizer`` to build the tokenizer from its slow files, which
    is tried only when transformers is already installed. It is never imported
    otherwise: transformers decides once, at import, whether PyTorch is present, so
    importing it before the GLiNER auto-installer adds torch leaves gliner2 unable to
    load its model in this process (SDK-810).
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    try:
        tokenizer = Tokenizer.from_file(hf_hub_download(model, "tokenizer.json"))
    except Exception as error:
        if importlib.util.find_spec("transformers") is None:
            # name="transformers": the resolver's fallback warning then names the extra.
            raise ImportError(
                f"could not load tokenizer.json for {model!r} ({error}), and transformers, "
                "which can build the tokenizer from the repo's other files, is not installed",
                name="transformers",
            ) from error
        from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

        auto = AutoTokenizer.from_pretrained(model)
        return auto.tokenize, auto.num_special_tokens_to_add(pair=False)
    # Some repos store truncation and fixed-length padding in tokenizer.json
    # (sentence-transformers/all-MiniLM-L6-v2: both at 128), which encode() then
    # applies, so every text would count as 128 tokens. AutoTokenizer.tokenize
    # never applied them; counting must see the whole text, unpadded.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return (
        lambda text: tokenizer.encode(text, add_special_tokens=False).tokens,
        tokenizer.num_special_tokens_to_add(is_pair=False),
    )


def _declared_input_limit(model: str) -> int | None:
    """``model_max_length`` from the repo's ``tokenizer_config.json``, or None when the
    repo has no such file or declares no positive integer limit."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(model, "tokenizer_config.json")
    except EntryNotFoundError:
        return None
    limit = json.loads(Path(path).read_text(encoding="utf-8")).get("model_max_length")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return None
    return limit


class HuggingFaceTokenizer(TokenizerInterface):
    """
    Counts tokens with a HuggingFace repo's own tokenizer (see ``_load``).

    Public methods include:
    - extract_tokens
    - count_tokens
    - decode_single_token

    Instance variables include:
    - model: str
    - max_completion_tokens: int
    """

    def __init__(
        self,
        model: str,
        max_completion_tokens: int = 512,
    ) -> None:
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self._tokenize, self._special_tokens = _load(model)
        self._declared_limit = _declared_input_limit(model)

    def extract_tokens(self, text: str) -> list[Any]:
        """
        Extract tokens from the given text using the tokenizer.

        Parameters:
        -----------

            - text (str): The input text to be tokenized.

        Returns:
        --------

            - List[Any]: A list of tokens extracted from the input text.
        """
        return self._tokenize(text)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Parameters:
        -----------

            - text (str): The input text for which to count tokens.

        Returns:
        --------

            - int: The total number of tokens in the input text.
        """
        return len(self._tokenize(text))

    @property
    def model_input_limit(self) -> int | None:
        """The input limit the model's repo declares (``model_max_length`` in its
        ``tokenizer_config.json``), less the special tokens the model adds itself,
        since text is counted without them. None when the repo declares no limit.
        """
        if self._declared_limit is None:
            return None
        return self._declared_limit - self._special_tokens

    def decode_single_token(self, token: int) -> str:
        """
        Attempt to decode a single token from its encoding, which is not implemented in this
        tokenizer.

        Parameters:
        -----------

            - encoding (int): The integer encoding of the token to decode.
        """
        # HuggingFace tokenizer doesn't have the option to decode tokens
        raise NotImplementedError
