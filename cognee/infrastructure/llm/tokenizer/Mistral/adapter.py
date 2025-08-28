from typing import List, Any

from ..tokenizer_interface import TokenizerInterface


class MistralTokenizer(TokenizerInterface):
    """
    Tokenizes input text based on a specified model while adhering to a maximum token limit.
    The class implements the TokenizerInterface and provides methods to extract tokens,
    count them, and decode single tokens (although decoding is not supported). Public
    methods include:

    - extract_tokens(text: str)
    - count_tokens(text: str)
    - decode_single_token(encoding: int)

    Instance variables include:
    - model: str
    - max_completion_tokens: int
    - tokenizer: MistralTokenizer
    """

    def __init__(
        self,
        model: str,
        max_completion_tokens: int = 3072,
    ):
        self.model = model
        self.max_completion_tokens = max_completion_tokens

        # Import here to make it an optional dependency
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        self.tokenizer = MistralTokenizer.from_model(model)

    def extract_tokens(self, text: str) -> List[Any]:
        """
        Extracts tokens from the given text using the tokenizer model.

        Parameters:
        -----------

            - text (str): The input text from which to extract tokens.

        Returns:
        --------

            - List[Any]: A list of extracted tokens.
        """
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.tokens.tokenizers.base import Tokenized

        encoding: Tokenized = self.tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[UserMessage(role="user", content=text)],
                model=self.model,
            )
        )
        return encoding.tokens

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text.

        Parameters:
        -----------

            - text (str): The input text for which to count tokens.

        Returns:
        --------

            - int: The number of tokens in the given text.
        """
        return len(self.extract_tokens(text))

    def decode_single_token(self, encoding: int):
        """
        Attempt to decode a single token, although this functionality is not implemented and
        raises NotImplementedError.

        Parameters:
        -----------

            - encoding (int): The integer representation of the token to decode.
        """
        # Mistral tokenizer doesn't have the option to decode tokens
        raise NotImplementedError
