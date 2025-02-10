from typing import List, Any

from ..tokenizer_interface import TokenizerInterface


class MistralTokenizer(TokenizerInterface):
    def __init__(
        self,
        model: str,
        max_tokens: int = 3072,
    ):
        self.model = model
        self.max_tokens = max_tokens

        # Import here to make it an optional dependency
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        self.tokenizer = MistralTokenizer.from_model(model)

    def extract_tokens(self, text: str) -> List[Any]:
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
        Returns the number of tokens in the given text.
        Args:
            text: str

        Returns:
            number of tokens in the given text

        """
        return len(self.extract_tokens(text))

    def decode_single_token(self, encoding: int):
        # Mistral tokenizer doesn't have the option to decode tokens
        raise NotImplementedError
