import asyncio
import math
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.lexical_retriever import LexicalRetriever, tokenize_words
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS


@dataclass(frozen=True)
class _BM25Corpus:
    chunks: dict[str, list[str]]
    payloads: dict[str, dict]
    idf: dict[str, float]
    avg_chunk_length: float
    token_count: int
    expires_at: float


class BM25ChunksRetriever(LexicalRetriever):
    """
    Retriever that specializes LexicalRetriever to rank chunks with Okapi BM25.

    Corpus statistics (per-token IDF and average chunk length) are computed once during
    initialize() from the already-tokenized chunks, then read by the scorer. This keeps the
    in-memory model of LexicalRetriever and adds no dependency or persistence.
    """

    CACHE_TTL_SECONDS = 60.0
    CACHE_MAX_ENTRIES = 8
    CACHE_MAX_TOKENS = 1_000_000

    # Hybrid search constructs a retriever per query. Cache immutable corpus snapshots at
    # class scope so those instances can reuse graph reads, tokenization, and BM25 stats.
    # A threading lock protects metadata because ContextVar-backed dataset searches may be
    # driven by different event loops. concurrent.futures.Future lets concurrent async
    # callers coalesce a cache miss without binding the rendezvous to one event loop.
    _corpus_cache: "OrderedDict[tuple, _BM25Corpus]" = OrderedDict()
    _inflight_loads: dict[tuple, Future] = {}
    _cache_lock = threading.RLock()
    _global_generation = 0
    _dataset_generations: dict[str, int] = {}
    _cache_metrics = {"hits": 0, "misses": 0, "waits": 0, "evictions": 0}

    def __init__(
        self,
        top_k: int = 15,
        with_scores: bool = False,
        stop_words: Optional[list[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        node_name: Optional[list[str]] = None,
        node_name_filter_operator: str = "OR",
        use_cache: bool = False,
        cache_ttl_seconds: Optional[float] = None,
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
        node_name : list[str], optional
            NodeSet names used to restrict the lexical corpus before ranking.
        node_name_filter_operator : str
            ``AND`` requires every requested NodeSet; all other values use ``OR``.
        use_cache : bool
            Reuse a bounded corpus snapshot when a dataset context is active. Disabled by
            default so direct BM25 callers retain the existing fresh-read behavior.
        cache_ttl_seconds : float, optional
            Snapshot lifetime. Defaults to ``CACHE_TTL_SECONDS``.
        """
        if stop_words is None:
            self.stop_words = set(DEFAULT_STOP_WORDS)
        else:
            self.stop_words = {t.lower() for t in stop_words}
        self.k1 = k1
        self.b = b
        self.node_name = sorted({str(name) for name in (node_name or [])})
        self.node_name_filter_operator = node_name_filter_operator.upper()
        self.use_cache = bool(use_cache)
        self.cache_ttl_seconds = (
            self.CACHE_TTL_SECONDS
            if cache_ttl_seconds is None
            else max(0.0, float(cache_ttl_seconds))
        )
        self.cache_status = "not_initialized"

        # Corpus statistics, populated once by initialize().
        self.idf: dict[str, float] = {}
        self.avg_chunk_length: float = 0.0
        self._stats_built = False

        super().__init__(
            tokenizer=self._tokenizer,
            scorer=self._scorer,
            top_k=top_k,
            with_scores=with_scores,
            payload_filter=self._matches_node_scope,
        )

    def _tokenizer(self, text: str) -> list[str]:
        """Lowercases, splits on word characters (w+), filters stopwords."""
        return tokenize_words(text, self.stop_words)

    async def initialize(self):
        """Load chunks via the parent, then build BM25 corpus statistics once.

        The parent owns ``_initialized`` and returns early on re-entry, so a separate
        ``_stats_built`` guard is needed to avoid recomputing stats on a repeated call.
        """
        if self._initialized and self._stats_built:
            return

        dataset_id = current_dataset_id.get()
        if not self.use_cache or dataset_id is None or self.cache_ttl_seconds <= 0:
            self.cache_status = "bypass"
            await self._initialize_uncached()
            return

        cache_key = self._cache_key(str(dataset_id))
        now = time.monotonic()
        owner = False
        with self._cache_lock:
            self._prune_expired_locked(now)
            cached = self._corpus_cache.get(cache_key)
            if cached is not None:
                self._corpus_cache.move_to_end(cache_key)
                self._cache_metrics["hits"] += 1
                self.cache_status = "hit"
                self._apply_corpus(cached)
                return

            future = self._inflight_loads.get(cache_key)
            if future is None:
                future = Future()
                self._inflight_loads[cache_key] = future
                generation = self._generation_for(str(dataset_id))
                self._cache_metrics["misses"] += 1
                self.cache_status = "miss"
                owner = True
            else:
                generation = self._generation_for(str(dataset_id))
                self._cache_metrics["waits"] += 1
                self.cache_status = "wait"

        if not owner:
            corpus = await asyncio.shield(asyncio.wrap_future(future))
            self._apply_corpus(corpus)
            return

        try:
            await self._initialize_uncached()
            corpus = self._snapshot_corpus(time.monotonic() + self.cache_ttl_seconds)
        except BaseException as error:
            with self._cache_lock:
                if self._inflight_loads.get(cache_key) is future:
                    self._inflight_loads.pop(cache_key, None)
                if not future.done():
                    future.set_exception(error)
            raise

        with self._cache_lock:
            is_current_load = self._inflight_loads.get(cache_key) is future
            generation_is_current = generation == self._generation_for(str(dataset_id))
            if is_current_load:
                self._inflight_loads.pop(cache_key, None)
            if is_current_load and generation_is_current:
                # Do not retain a single oversized corpus. For normal corpora, enforce
                # both entry-count and token-count bounds so a handful of large datasets
                # cannot turn the process-wide optimization into unbounded memory growth.
                if corpus.token_count <= self.CACHE_MAX_TOKENS:
                    self._corpus_cache[cache_key] = corpus
                    self._corpus_cache.move_to_end(cache_key)
                    self._enforce_cache_bounds_locked()
            if not future.done():
                future.set_result(corpus)

    async def _initialize_uncached(self) -> None:
        await super().initialize()
        if not self._stats_built:
            self._build_corpus_stats()
            self._stats_built = True

    def _matches_node_scope(self, document: dict) -> bool:
        if not self.node_name:
            return True
        belongs_to_set = document.get("belongs_to_set")
        if not isinstance(belongs_to_set, list):
            return False
        payload_sets = {str(name) for name in belongs_to_set}
        requested_sets = set(self.node_name)
        if self.node_name_filter_operator == "AND":
            return requested_sets.issubset(payload_sets)
        return bool(payload_sets & requested_sets)

    def _cache_key(self, dataset_id: str) -> tuple:
        return (
            dataset_id,
            tuple(self.node_name),
            self.node_name_filter_operator,
            tuple(sorted(self.stop_words)),
        )

    def _snapshot_corpus(self, expires_at: float) -> _BM25Corpus:
        return _BM25Corpus(
            chunks=dict(self.chunks),
            payloads=dict(self.payloads),
            idf=dict(self.idf),
            avg_chunk_length=self.avg_chunk_length,
            token_count=sum(len(tokens) for tokens in self.chunks.values()),
            expires_at=expires_at,
        )

    def _apply_corpus(self, corpus: _BM25Corpus) -> None:
        self.chunks = dict(corpus.chunks)
        self.payloads = dict(corpus.payloads)
        self.idf = dict(corpus.idf)
        self.avg_chunk_length = corpus.avg_chunk_length
        self._initialized = True
        self._stats_built = True

    @classmethod
    def _generation_for(cls, dataset_id: str) -> tuple[int, int]:
        return cls._global_generation, cls._dataset_generations.get(dataset_id, 0)

    @classmethod
    def _prune_expired_locked(cls, now: float) -> None:
        expired = [key for key, corpus in cls._corpus_cache.items() if corpus.expires_at <= now]
        for key in expired:
            cls._corpus_cache.pop(key, None)

    @classmethod
    def _enforce_cache_bounds_locked(cls) -> None:
        cached_tokens = sum(corpus.token_count for corpus in cls._corpus_cache.values())
        while cls._corpus_cache and (
            len(cls._corpus_cache) > cls.CACHE_MAX_ENTRIES or cached_tokens > cls.CACHE_MAX_TOKENS
        ):
            _, evicted = cls._corpus_cache.popitem(last=False)
            cached_tokens -= evicted.token_count
            cls._cache_metrics["evictions"] += 1

    @classmethod
    def invalidate_cache(cls, dataset_id: Optional[str] = None) -> int:
        """Invalidate cached lexical corpora, optionally for one dataset.

        In-flight loads are detached as well. They may finish for their existing callers,
        but generation checks prevent stale snapshots from repopulating the cache.
        """
        with cls._cache_lock:
            if dataset_id is None:
                removed = len(cls._corpus_cache)
                cls._corpus_cache.clear()
                cls._inflight_loads.clear()
                cls._dataset_generations.clear()
                cls._global_generation += 1
                return removed

            normalized_id = str(dataset_id)
            cache_keys = [key for key in cls._corpus_cache if key[0] == normalized_id]
            inflight_keys = [key for key in cls._inflight_loads if key[0] == normalized_id]
            for key in cache_keys:
                cls._corpus_cache.pop(key, None)
            for key in inflight_keys:
                cls._inflight_loads.pop(key, None)
            cls._dataset_generations[normalized_id] = (
                cls._dataset_generations.get(normalized_id, 0) + 1
            )
            return len(cache_keys)

    @classmethod
    def cache_info(cls) -> dict[str, int]:
        with cls._cache_lock:
            return {
                **cls._cache_metrics,
                "entries": len(cls._corpus_cache),
                "inflight": len(cls._inflight_loads),
                "max_entries": cls.CACHE_MAX_ENTRIES,
                "cached_tokens": sum(corpus.token_count for corpus in cls._corpus_cache.values()),
                "max_tokens": cls.CACHE_MAX_TOKENS,
            }

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
