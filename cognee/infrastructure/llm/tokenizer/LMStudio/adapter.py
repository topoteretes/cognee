from typing import List, Any
import logging
import re

from ..tokenizer_interface import TokenizerInterface

logger = logging.getLogger(__name__)

class FallbackTokenizer:
    """A simple fallback tokenizer that splits text by whitespace and punctuation.

    This is used when tiktoken is not available. It's not as accurate as tiktoken
    but provides basic tokenization functionality.
    """

    def __init__(self):
        self.pattern = re.compile(r'\s+|[,.!?;:"]')

    def encode(self, text: str) -> List[int]:
        """Split text into tokens and assign arbitrary IDs."""
        tokens = [t for t in self.pattern.split(text) if t]
        # Use hash of token as a simple ID
        return [hash(token) % 100000 for token in tokens]

    def decode(self, tokens: List[int]) -> str:
        """This is a stub - can't actually decode with this tokenizer."""
        return "[Decoding not supported with fallback tokenizer]"

class LMStudioTokenizer(TokenizerInterface):
    """Tokenizer for LM Studio models.

    LM Studio uses tiktoken for tokenization, which is compatible with many models.
    This tokenizer attempts to use the appropriate encoding for the model,
    falling back to cl100k_base (used by GPT-4 and many other models) if the
    specific encoding is not available.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
    ):
        """Initialize the LM Studio tokenizer.

        Args:
            model: The model identifier
            max_tokens: Maximum number of tokens (default: 2048)
        """
        self.model = model
        self.max_tokens = max_tokens

        # Import here to make it an optional dependency
        try:
            import tiktoken

            # Try to get the encoding for the model, fallback to cl100k_base
            try:
                self.tokenizer = tiktoken.encoding_for_model(model)
                logger.info(f"Using tiktoken encoding for model: {model}")
            except KeyError:
                # Fallback to cl100k_base (used by GPT-4 and many other models)
                logger.info(f"No specific encoding found for {model}, using cl100k_base")
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed. Using fallback tokenizer.")
            self.tokenizer = FallbackTokenizer()

    def extract_tokens(self, text: str) -> List[Any]:
        """Extract tokens from text."""
        return self.tokenizer.encode(text)

    def count_tokens(self, text: str) -> int:
        """
        Returns the number of tokens in the given text.
        Args:
            text: str

        Returns:
            number of tokens in the given text
        """
        return len(self.tokenizer.encode(text))

    def decode_single_token(self, encoding: int):
        """Decode a single token."""
        return self.tokenizer.decode([encoding])