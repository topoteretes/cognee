"""The word tokenizer every lexical engine agrees on.

Lower-case, split on word characters, drop stop words. The in-memory scorer,
the Postgres posting index in ``simple`` mode and the LanceDB index with
stemming off all produce these tokens, so the same corpus ranks the same way
on every backend.
"""

import re


def tokenize_words(text: str, stop_words: set[str] | None = None) -> list[str]:
    """Lowercase, split on word characters, and drop any stop words."""
    tokens = re.findall(r"\w+", text.lower())
    if not stop_words:
        return tokens
    return [token for token in tokens if token not in stop_words]
