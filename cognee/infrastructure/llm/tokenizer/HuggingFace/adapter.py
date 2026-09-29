import importlib.util
from collections.abc import Callable
from typing import Any

from ..tokenizer_interface import TokenizerInterface


def _load_tokenize(model: str) -> Callable[[str], list[str]]:
    """Return ``text -> tokens`` for a HuggingFace repo, without special tokens.

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
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model).tokenize
    # Some repos save truncation and fixed-length padding in tokenizer.json
    # (sentence-transformers/all-MiniLM-L6-v2: both at 128), which encode() then
    # applies, so every text would count as 128 tokens. AutoTokenizer.tokenize
    # never applied them; counting must see the whole text, unpadded.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return lambda text: tokenizer.encode(text, add_special_tokens=False).tokens


class HuggingFaceTokenizer(TokenizerInterface):
    """
    Counts tokens with a HuggingFace repo's own tokenizer (see ``_load_tokenize``).

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
        self._tokenize = _load_tokenize(model)

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
        """The input limit the model's repo declares (``model_max_length``), less the
        special tokens the model adds itself, since text is counted without them.
        None when the repo declares no limit (transformers then substitutes a
        placeholder, so the declared value is read from ``init_kwargs``).
        """
        limit = self.tokenizer.init_kwargs.get("model_max_length")
        if not isinstance(limit, int) or limit <= 0:
            return None
        return limit - self.tokenizer.num_special_tokens_to_add(pair=False)

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
