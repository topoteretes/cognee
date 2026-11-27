"""Answer Relevance Evaluator for TrustRAG.

Assesses whether a generated answer actually addresses the user's question,
producing an independent diagnostic signal distinct from the factual trust score.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_REFUSAL_PATTERNS = [
    re.compile(r"insufficient\s+(?:evidence|information|data|context)", re.IGNORECASE),
    re.compile(r"not\s+(?:contain\s+sufficient|provide\s+sufficient|have\s+sufficient)", re.IGNORECASE),
    re.compile(
        r"(?:is|are|was|were)\s+not\s+(?:disclosed|mentioned|specified|stated|provided)\s+in\s+the\s+(?:provided\s+|supplied\s+)?evidence",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:does|do|did)\s+not\s+(?:disclose|mention|specify|state|contain|provide)",
        re.IGNORECASE,
    ),
    re.compile(
        r"unable\s+to\s+(?:determine|answer|find|verify)\s+(?:from|based\s+on)\s+the\s+(?:provided\s+|supplied\s+)?evidence",
        re.IGNORECASE,
    ),
    re.compile(
        r"cannot\s+be\s+(?:determined|answered|found|verified)\s+(?:from|based\s+on)\s+the\s+(?:provided\s+|supplied\s+)?evidence",
        re.IGNORECASE,
    ),
    re.compile(
        r"no\s+(?:evidence|information|record|mention)\s+(?:found|provided|available|exists)",
        re.IGNORECASE,
    ),
    re.compile(
        r"based\s+on\s+the\s+(?:provided\s+|supplied\s+)?evidence,?\s+there\s+is\s+no",
        re.IGNORECASE,
    ),
    re.compile(
        r"the\s+provided\s+evidence\s+does\s+not\s+(?:contain|mention|state|disclose)",
        re.IGNORECASE,
    ),
]


def is_refusal_text(text: str) -> bool:
    """Check if the provided text matches standard refusal or insufficient-evidence patterns."""
    if not text or not text.strip():
        return False
    cleaned = text.strip()
    return any(p.search(cleaned) is not None for p in _REFUSAL_PATTERNS)


_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "please", "provide", "same",
    "shall", "shan't", "she", "should", "shouldn't", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "won't", "would", "wouldn't", "you",
    "your", "yours", "yourself", "yourselves",
}


class AnswerRelevanceReport(BaseModel):
    """Diagnostic report assessing answer relevance to the user's question."""

    query: str = Field(description="The original user question")
    answer: str = Field(description="The evaluated answer text")
    is_relevant: bool = Field(description="True if the answer is determined to address the question")
    relevance_score: float = Field(description="Normalized relevance score in [0.0, 1.0]")
    relevance_threshold: float = Field(description="Decision threshold applied")
    is_refusal: bool = Field(default=False, description="True if answer is a responsive refusal")
    method: str = Field(description="Assessment method used ('responsive_refusal', 'embedding_hybrid', 'lexical_coverage')")
    rationale: str = Field(description="Human-readable diagnostic summary")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sub-score details (lexical_coverage, semantic_similarity, etc.)",
    )


def extract_substantive_tokens(text: str) -> List[str]:
    """Extract lowercased substantive content words from text, filtering stopwords."""
    words = re.findall(r"\b[A-Za-z0-9_-]+\b", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def compute_lexical_query_coverage(query: str, answer: str) -> Tuple[float, List[str], List[str]]:
    """Compute what fraction of substantive query concepts appear in the answer."""
    q_tokens = extract_substantive_tokens(query)
    if not q_tokens:
        return 1.0, [], []

    ans_lower = answer.lower()
    matched = []
    missing = []
    for tok in q_tokens:
        if re.search(r"\b" + re.escape(tok) + r"\b", ans_lower) or tok in ans_lower:
            matched.append(tok)
        else:
            missing.append(tok)

    coverage = len(matched) / len(q_tokens)
    return round(coverage, 4), matched, missing


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors clamped to [0.0, 1.0]."""
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(1.0, sim))


def check_answer_relevance(
    query: str,
    answer: str,
    threshold: float = 0.50,
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
) -> AnswerRelevanceReport:
    """Evaluate whether answer addresses question without altering the trust score."""
    cleaned_query = (query or "").strip()
    cleaned_answer = (answer or "").strip()

    if not cleaned_query:
        return AnswerRelevanceReport(
            query=cleaned_query,
            answer=cleaned_answer,
            is_relevant=False,
            relevance_score=0.0,
            relevance_threshold=threshold,
            is_refusal=False,
            method="empty_query",
            rationale="Query is empty.",
        )

    if not cleaned_answer:
        return AnswerRelevanceReport(
            query=cleaned_query,
            answer=cleaned_answer,
            is_relevant=False,
            relevance_score=0.0,
            relevance_threshold=threshold,
            is_refusal=False,
            method="empty_answer",
            rationale="Answer is empty.",
        )

    if is_refusal_text(cleaned_answer):
        return AnswerRelevanceReport(
            query=cleaned_query,
            answer=cleaned_answer,
            is_relevant=True,
            relevance_score=1.0,
            relevance_threshold=threshold,
            is_refusal=True,
            method="responsive_refusal",
            rationale="Answer is a responsive refusal appropriately identifying insufficient evidence.",
            details={"is_refusal": True},
        )

    lex_cov, matched, missing = compute_lexical_query_coverage(cleaned_query, cleaned_answer)

    sem_score: Optional[float] = None
    if embedding_fn is not None:
        try:
            q_emb = embedding_fn(cleaned_query)
            a_emb = embedding_fn(cleaned_answer)
            sem_score = _cosine_similarity(q_emb, a_emb)
        except Exception as exc:
            logger.debug("Embedding computation failed for relevance check: %s", exc)

    if sem_score is not None:
        final_score = round(0.5 * sem_score + 0.5 * lex_cov, 4)
        method = "embedding_hybrid"
        details = {
            "semantic_similarity": round(sem_score, 4),
            "lexical_coverage": lex_cov,
            "matched_tokens": matched,
            "missing_tokens": missing,
        }
    else:
        final_score = lex_cov
        method = "lexical_coverage"
        details = {
            "lexical_coverage": lex_cov,
            "matched_tokens": matched,
            "missing_tokens": missing,
        }

    is_rel = final_score >= threshold
    if is_rel:
        rationale = (
            f"Answer addresses query (relevance score {final_score:.3f} >= {threshold:.3f}, "
            f"matched {len(matched)}/{len(matched) + len(missing)} key query terms via {method})."
        )
    else:
        rationale = (
            f"Answer may not adequately address query (relevance score {final_score:.3f} < {threshold:.3f}, "
            f"missing query terms {missing[:5]} via {method})."
        )

    return AnswerRelevanceReport(
        query=cleaned_query,
        answer=cleaned_answer,
        is_relevant=is_rel,
        relevance_score=final_score,
        relevance_threshold=threshold,
        is_refusal=False,
        method=method,
        rationale=rationale,
        details=details,
    )

