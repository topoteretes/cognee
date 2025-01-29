from typing import List, Any, Union

from ..tokenizer_interface import TokenizerInterface


class GeminiTokenizer(TokenizerInterface):
    def __init__(
        self,
        model: str,
        max_tokens: int = 3072,
    ):
        self.model = model
        self.max_tokens = max_tokens

        # Get LLM API key from config
        from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
        from cognee.infrastructure.llm.config import get_llm_config

        config = get_embedding_config()
        llm_config = get_llm_config()

        import google.generativeai as genai

        genai.configure(api_key=config.embedding_api_key or llm_config.llm_api_key)

    def extract_tokens(self, text: str) -> List[Any]:
        raise NotImplementedError

    def decode_single_token(self, encoding: int):
        # Gemini tokenizer doesn't have the option to decode tokens
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """
        Returns the number of tokens in the given text.
        Args:
            text: str

        Returns:
            number of tokens in the given text

        """
        import google.generativeai as genai

        return len(genai.embed_content(model=f"models/{self.model}", content=text))
