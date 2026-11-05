"""Tokenizer, stemmer and BM25 index behind the CODE ``search`` and ``ask`` operations.

Pure text processing: nothing here reads the graph, calls a model or touches
the network. The CODE retriever feeds the index every fact's qualified name,
docstring and file path (plus each flow's summary) and asks which of them best
match a question. Queries and documents go through the same tokenizer, so
``validateToken``, ``validate_token`` and "validating tokens" all meet on the
terms ``valid`` and ``token``.
"""

from __future__ import annotations

import math
import re
from array import array
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

# English function words and question words, plus the filler a question about
# a codebase carries ("how does checkout work in this repo?"). Matched before
# stemming, against the lowercased token.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "work",
        "works",
        "happen",
        "happens",
        "deal",
        "deals",
        "code",
        "codebase",
        "repo",
        "repository",
    }
)

# Tried longest first (sorted() is stable, so equal lengths keep this order);
# the first suffix that leaves a stem of at least four characters is stripped.
_SUFFIXES = tuple(
    sorted(
        (
            "izations",
            "ization",
            "ations",
            "ation",
            "ators",
            "ator",
            "ating",
            "ates",
            "ated",
            "izing",
            "izers",
            "izer",
            "izes",
            "ized",
            "ize",
            "ate",
            "ings",
            "ing",
            "ions",
            "ion",
            "ers",
            "ors",
            "ies",
            "ed",
            "es",
            "er",
            "or",
            "ly",
            "s",
            "e",
        ),
        key=len,
        reverse=True,
    )
)
_MIN_STEM_LENGTH = 4
_MIN_TOKEN_LENGTH = 2
# Runs of letters and digits; every other character (``_ . / - :`` and
# whitespace included) separates words.
_WORD = re.compile(r"[^\W_]+")

FIELD_WEIGHTS = {"name": 3.0, "doc": 2.0, "file": 1.0, "summary": 1.0}
FIELDS = tuple(FIELD_WEIGHTS)
_WEIGHTS = tuple(FIELD_WEIGHTS.values())
# The matched fields of a document, by the bit mask search() keeps: built
# once rather than for each of a common term's many matches.
_MATCHED_FIELDS = tuple(
    tuple(field for position, field in enumerate(FIELDS) if mask & (1 << position))
    for mask in range(1 << len(FIELDS))
)
# Per-field term frequencies are packed eight bits each into one 32-bit
# value per posting, which keeps a large repository's index compact.
_TF_BITS = 8
_TF_MAX = (1 << _TF_BITS) - 1
_UINT32 = "I" if array("I").itemsize >= 4 else "L"


def split_identifier(word: str) -> list[str]:
    """Split one alphanumeric run at camelCase and acronym boundaries.

    ``validateToken`` -> ``validate``, ``Token``; ``HTTPServer`` -> ``HTTP``,
    ``Server``; ``sha256Hash`` -> ``sha256``, ``Hash``. Digits stay with the
    letters before them.
    """
    parts = []
    start = 0
    for index in range(1, len(word)):
        char = word[index]
        if not char.isupper():
            continue
        previous = word[index - 1]
        follows_lower = previous.islower() or previous.isdigit()
        ends_acronym = previous.isupper() and index + 1 < len(word) and word[index + 1].islower()
        if follows_lower or ends_acronym:
            parts.append(word[start:index])
            start = index
    parts.append(word[start:])
    return parts


def stem(token: str) -> str:
    """Strip one inflectional suffix so related word forms share a stem.

    Deliberately light and deterministic: validate, validates, validated,
    validating, validation and validator all become ``valid``; ``class`` and
    ``status`` stay whole (no ``s`` is stripped after s, u or i).
    """
    for suffix in _SUFFIXES:
        if not token.endswith(suffix):
            continue
        base = token[: -len(suffix)]
        if suffix == "s" and base[-1:] in {"s", "u", "i"}:
            continue
        if suffix == "ies":
            base += "y"
        if len(base) >= _MIN_STEM_LENGTH:
            return base
    return token


@lru_cache(maxsize=1 << 16)
def _run_terms(run: str) -> tuple[str, ...]:
    # Identifier parts repeat across names, docstrings and paths, so the
    # per-run work is cached: it is most of the cost of indexing a repository.
    terms = []
    for part in split_identifier(run):
        token = part.lower()
        if len(token) < _MIN_TOKEN_LENGTH or token in STOPWORDS:
            continue
        terms.append(stem(token))
    return tuple(terms)


def tokenize(text: str | None) -> list[str]:
    """Searchable terms of a text, in order and with repeats."""
    if not text:
        return []
    return [term for run in _WORD.findall(text) for term in _run_terms(run)]


def query_terms(text: str | None) -> list[str]:
    """The distinct terms of a query, in the order they first appear."""
    return list(dict.fromkeys(tokenize(text)))


@dataclass(frozen=True)
class SearchMatch:
    """One document's BM25 score and the fields a query term matched in."""

    score: float
    fields: tuple[str, ...]


class CodeSearchIndex:
    """Field-weighted BM25 over code-graph documents.

    Each document is a key plus up to four text fields (``name``, ``doc``,
    ``file``, ``summary``). A term scores per field with BM25 (k1, b), the
    field's length normalised against that field's average over the
    documents that have it; field scores are weighted (name 3, doc 2,
    file 1, summary 1) and summed. Document frequency counts documents,
    whichever of their fields holds the term, and
    idf = ln(1 + (N - df + 0.5) / (df + 0.5)).
    """

    def __init__(
        self,
        documents: Iterable[tuple[str, Mapping[str, str | None]]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.keys: list[str] = []
        self._lengths = tuple(array(_UINT32) for _ in FIELDS)
        self._postings: dict[str, tuple[array, array]] = {}
        for key, fields in documents:
            document = len(self.keys)
            self.keys.append(key)
            counts = [Counter(tokenize(fields.get(field))) for field in FIELDS]
            for lengths, field_counts in zip(self._lengths, counts):
                lengths.append(sum(field_counts.values()))
            for term in sorted(set().union(*counts)):
                packed = 0
                for position, field_counts in enumerate(counts):
                    packed |= min(field_counts.get(term, 0), _TF_MAX) << (_TF_BITS * position)
                postings = self._postings.get(term)
                if postings is None:
                    postings = self._postings[term] = (array(_UINT32), array(_UINT32))
                postings[0].append(document)
                postings[1].append(packed)
        self._average_lengths = tuple(self._average(lengths) for lengths in self._lengths)

    @staticmethod
    def _average(lengths: array) -> float:
        present = [length for length in lengths if length]
        return sum(present) / len(present) if present else 1.0

    def __len__(self) -> int:
        return len(self.keys)

    def idf(self, term: str) -> float:
        postings = self._postings.get(term)
        frequency = len(postings[0]) if postings else 0
        return math.log(1 + (len(self.keys) - frequency + 0.5) / (frequency + 0.5))

    def search(self, terms: Sequence[str]) -> dict[str, SearchMatch]:
        """Score every document that holds at least one of the terms."""
        scores: dict[int, float] = {}
        matched: dict[int, int] = {}
        k1, b = self.k1, self.b
        for term in dict.fromkeys(terms):
            postings = self._postings.get(term)
            if postings is None:
                continue
            idf = self.idf(term)
            for document, packed in zip(*postings):
                score = 0.0
                for position, weight in enumerate(_WEIGHTS):
                    frequency = (packed >> (_TF_BITS * position)) & _TF_MAX
                    if not frequency:
                        continue
                    length = self._lengths[position][document]
                    norm = k1 * (1 - b + b * length / self._average_lengths[position])
                    score += weight * idf * frequency * (k1 + 1) / (frequency + norm)
                    matched[document] = matched.get(document, 0) | (1 << position)
                scores[document] = scores.get(document, 0.0) + score
        return {
            self.keys[document]: SearchMatch(score=score, fields=_MATCHED_FIELDS[matched[document]])
            for document, score in scores.items()
        }
