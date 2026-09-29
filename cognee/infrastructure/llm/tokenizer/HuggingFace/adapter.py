from typing import Any

from ..tokenizer_interface import TokenizerInterface


class HuggingFaceTokenizer(TokenizerInterface):
    """
    Implements a tokenizer using the Hugging Face Transformers library.

    Public methods include:
    - extract_tokens
    - count_tokens
    - decode_single_token

    Instance variables include:
    - model: str
    - max_completion_tokens: int
    - tokenizer: AutoTokenizer
    """

    def __init__(
        self,
        model: str,
        max_completion_tokens: int = 512,
    ) -> None:
        self.model = model
        self.max_completion_tokens = max_completion_tokens

        # Import here to make it an optional dependency
        from transformers import AutoTokenizer  # ty:ignore[unresolved-import]

        self.tokenizer = AutoTokenizer.from_pretrained(model)

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
        tokens = self.tokenizer.tokenize(text)
        return tokens

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
        return len(self.tokenizer.tokenize(text))

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
