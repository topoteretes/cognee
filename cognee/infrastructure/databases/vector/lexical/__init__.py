"""Lexical (BM25) indexing shared by the vector backends.

A vector backend may keep a second index kind on a collection next to the
embedding index: a keyword index scored with BM25. ``DataPoint.metadata``
declares the fields with ``lexical_fields`` and ``index_data_points`` drives
the three ``VectorDBInterface`` methods (``create_lexical_index``,
``index_lexical_data_points``, ``lexical_search``). What lives here is the
part every engine shares: the tokenizer, the stop words, the passage split
and the language setting.
"""

from .passages import DEFAULT_PASSAGE_TOKENS, split_passages
from .stop_words import DEFAULT_STOP_WORDS
from .tokenize import tokenize_words

LEXICAL_SEARCH_LANGUAGES = ("simple", "english")

__all__ = [
    "DEFAULT_PASSAGE_TOKENS",
    "DEFAULT_STOP_WORDS",
    "LEXICAL_SEARCH_LANGUAGES",
    "split_passages",
    "tokenize_words",
]
