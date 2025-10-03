from typing import List, Any

from ..tokenizer_interface import TokenizerInterface


# NOTE: DEPRECATED as to count tokens you need to send an API request to Google it is too slow to use with Cognee
class GeminiTokenizer(TokenizerInterface):
    """
    Implements a tokenizer interface for the Gemini model, managing token extraction and
    counting.

    Public methods:
    - extract_tokens
    - decode_single_token
    - count_tokens
    """

    def __init__(
        self,
        llm_model: str,
        max_completion_tokens: int = 3072,
    ):
        self.llm_model = llm_model
        self.max_completion_tokens = max_completion_tokens

        # Get LLM API key from config
        from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
        from cognee.infrastructure.llm.config import (
            get_llm_config,
        )

        llm_config = get_llm_config()

        from google import genai

        self.client = genai.Client(api_key=llm_config.llm_api_key)

    def extract_tokens(self, text: str) -> List[Any]:
        """
        Raise NotImplementedError when called, as this method should be implemented in a
        subclass.

        Parameters:
        -----------

            - text (str): Input text from which to extract tokens.
        """
        raise NotImplementedError

    def decode_single_token(self, encoding: int):
        """
        Raise NotImplementedError when called, as Gemini tokenizer does not support decoding of
        tokens.

        Parameters:
        -----------

            - encoding (int): The token encoding to decode.
        """
        # Gemini tokenizer doesn't have the option to decode tokens
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """
        Returns the number of tokens in the given text.

        This method utilizes the Google Generative AI API to embed the content and count the
        tokens.

        Parameters:
        -----------

            - text (str): Input text for which to count tokens.

        Returns:
        --------

            - int: The number of tokens in the given text.
        """

        tokens_response = self.client.models.count_tokens(model=self.llm_model, contents=text)

        return tokens_response.total_tokens
