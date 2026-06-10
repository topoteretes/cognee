import math
from collections import Counter
from typing import Optional

from cognee.modules.retrieval.lexical_retriever import LexicalRetriever, tokenize_words
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS


class BM25ChunksRetriever(LexicalRetriever):
    """
    Retriever that specializes LexicalRetriever to rank chunks with Okapi BM25.

    Corpus statistics (per-token IDF and average chunk length) are computed once during
    initialize() from the already-tokenized chunks, then read by the scorer. This keeps the
    in-memory model of LexicalRetriever and adds no dependency or persistence.
    """

    def __init__(
        self,
        top_k: int = 15,
        with_scores: bool = False,
        stop_words: Optional[list[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Parameters
        ----------
        top_k : int
            Number of top results to return.
        with_scores : bool
            If True, return (payload, score) pairs. Otherwise, only payloads.
        stop_words : list[str], optional
            Tokens to filter out during tokenization. Defaults to DEFAULT_STOP_WORDS;
            pass an empty list to disable stopword filtering.
        k1 : float
            BM25 term-frequency saturation parameter.
        b : float
            BM25 length-normalization parameter.
        """
        if stop_words is None:
            self.stop_words = set(DEFAULT_STOP_WORDS)
        else:
            self.stop_words = {t.lower() for t in stop_words}
        self.k1 = k1
        self.b = b

        # Corpus statistics, populated once by initialize().
        self.idf: dict[str, float] = {}
        self.avg_chunk_length: float = 0.0
        self._stats_built = False

        super().__init__(
            tokenizer=self._tokenizer, scorer=self._scorer, top_k=top_k, with_scores=with_scores
        )

    def _tokenizer(self, text: str) -> list[str]:
        """Lowercases, splits on word characters (w+), filters stopwords."""
        return tokenize_words(text, self.stop_words)

    async def initialize(self):
        """Load chunks via the parent, then build BM25 corpus statistics once.

        The parent owns ``_initialized`` and returns early on re-entry, so a separate
        ``_stats_built`` guard is needed to avoid recomputing stats on a repeated call.
        """
        await super().initialize()
        if self._stats_built:
            return
        self._build_corpus_stats()
        self._stats_built = True

    def _build_corpus_stats(self):
        """Compute average chunk length and per-token IDF from the tokenized chunks."""
        total_chunks = len(self.chunks)
        document_frequency: Counter = Counter()
        total_length = 0
        for tokens in self.chunks.values():
            total_length += len(tokens)
            for token in set(tokens):
                document_frequency[token] += 1

        self.avg_chunk_length = total_length / total_chunks if total_chunks else 0.0
        self.idf = {
            token: math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))
            for token, df in document_frequency.items()
        }

    def _scorer(self, query_tokens: list[str], chunk_tokens: list[str]) -> float:
        """Okapi BM25 score of a chunk against the query, summed over unique query terms."""
        if not query_tokens or not chunk_tokens or self.avg_chunk_length == 0:
            return 0.0

        term_frequencies = Counter(chunk_tokens)
        length_norm = self.k1 * (1 - self.b + self.b * len(chunk_tokens) / self.avg_chunk_length)

        score = 0.0
        for token in set(query_tokens):
            tf = term_frequencies.get(token, 0)
            if tf == 0:
                continue
            idf = self.idf.get(token, 0.0)
            score += idf * (tf * (self.k1 + 1)) / (tf + length_norm)
        return score
