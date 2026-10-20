"""Common stop words for text processing.

The list moved to ``cognee.infrastructure.databases.vector.lexical`` so the
backend lexical indexes share it; this module keeps the old import path.
"""

from cognee.infrastructure.databases.vector.lexical.stop_words import DEFAULT_STOP_WORDS

__all__ = ["DEFAULT_STOP_WORDS"]
