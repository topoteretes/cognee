from cognee.modules.retrieval.lexical_retriever import LexicalRetriever
import re
from collections import Counter
from typing import Optional


class JaccardChunksRetriever(LexicalRetriever):
    """
    Retriever that specializes LexicalRetriever to use Jaccard similarity.
    """

    def __init__(
        self,
        top_k: int = 10,
        with_scores: bool = False,
        stop_words: Optional[list[str]] = None,
        multiset_jaccard: bool = False,
    ):
        """
        Parameters
        ----------
        top_k : int
            Number of top results to return.
        with_scores : bool
            If True, return (payload, score) pairs. Otherwise, only payloads.
        stop_words : list[str], optional
            List of tokens to filter out.
        multiset_jaccard : bool
            If True, use multiset Jaccard (frequency aware).
        """
        self.stop_words = {t.lower() for t in stop_words} if stop_words else set()
        self.multiset_jaccard = multiset_jaccard

        super().__init__(
            tokenizer=self._tokenizer, scorer=self._scorer, top_k=top_k, with_scores=with_scores
        )

    def _tokenizer(self, text: str) -> list[str]:
        """
        Tokenizer: lowercases, splits on word characters (w+), filters stopwords.
        """
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if t not in self.stop_words]

    def _scorer(self, query_tokens: list[str], chunk_tokens: list[str]) -> float:
        """
        Jaccard similarity scorer.
        - If multiset_jaccard=True, uses frequency-aware Jaccard.
        - Otherwise, normal set Jaccard.
        """
        if self.multiset_jaccard:
            q_counts, c_counts = Counter(query_tokens), Counter(chunk_tokens)
            numerator = sum(min(q_counts[t], c_counts[t]) for t in set(q_counts) | set(c_counts))
            denominator = sum(max(q_counts[t], c_counts[t]) for t in set(q_counts) | set(c_counts))
            return numerator / denominator if denominator else 0.0
        else:
            q_set, c_set = set(query_tokens), set(chunk_tokens)
            if not q_set or not c_set:
                return 0.0
            return len(q_set & c_set) / len(q_set | c_set)
