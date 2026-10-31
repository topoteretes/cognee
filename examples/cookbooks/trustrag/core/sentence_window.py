"""Sentence-Window Verification Engine.

Provides localized premise extraction within long evidence chunks to prevent
anaphora and dilution failures while strictly controlling false-entailment risk.
"""

from __future__ import annotations

from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ABBREVIATIONS = {
    "u.s.", "sec.", "inc.", "corp.", "co.", "ltd.", "llc.", "p.b.c.", "no.", "nos.",
    "vol.", "v.", "vs.", "al.", "etc.", "e.g.", "i.e.", "jan.", "feb.", "mar.",
    "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
    "art.", "para.", "subpara.", "item", "rule", "reg.", "stat.", "cf.",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "in", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
}


class SentenceWindowMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class WindowAggregationStrategy(str, Enum):
    QUOTE_ANCHORED = "quote_anchored"
    MAX = "max"


class SentenceWindowDetail(BaseModel):
    """Detailed score distribution for an individual sentence window."""

    window_index: int
    start_sentence_idx: int
    end_sentence_idx: int
    window_text: str
    has_quote_anchor: bool = False
    anchor_overlap_score: float = 0.0
    entailment_prob: float = 0.0
    neutral_prob: float = 0.0
    contradiction_prob: float = 0.0
    verdict: str = "neutral"


class SentenceWindowResult(BaseModel):
    """Container for sentence-window verification diagnostic and outcome."""

    mode: str = "off"
    original_chunk_verdict: str
    window_verdict: str
    final_verdict: str
    verdict_changed: bool = False
    change_direction: Optional[str] = None
    aggregation_strategy: str = "quote_anchored"
    total_sentences: int = 0
    total_windows: int = 0
    evaluated_windows: int = 0
    winning_window: Optional[SentenceWindowDetail] = None
    all_window_details: List[SentenceWindowDetail] = Field(default_factory=list)
    safety_override_applied: bool = False
    safety_override_reason: Optional[str] = None


def split_chunk_into_sentences(text: str) -> List[str]:
    """Segment a long chunk into sentences respecting abbreviations, numbers, and citations."""
    cleaned = text.strip()
    if not cleaned:
        return []

    raw_tokens = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])", cleaned)
    sentences: List[str] = []
    buffer = ""

    for tok in raw_tokens:
        tok_str = tok.strip()
        if not tok_str:
            continue
        if buffer:
            combined = buffer + " " + tok_str
        else:
            combined = tok_str

        last_word = combined.split()[-1].lower() if combined.split() else ""
        if last_word in _ABBREVIATIONS or re.search(r"\b(?:item|rule|sec)\s+\d+\.$", combined, re.I):
            buffer = combined
            continue

        sentences.append(combined)
        buffer = ""

    if buffer:
        if sentences:
            sentences[-1] = sentences[-1] + " " + buffer
        else:
            sentences.append(buffer)

    return sentences if sentences else [cleaned]


def create_sentence_windows(
    sentences: List[str],
    window_size: int = 3,
    stride: int = 1,
) -> List[Dict[str, Any]]:
    """Generate sliding sentence windows with start/end indices."""
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [
            {
                "window_index": 0,
                "start_sentence_idx": 0,
                "end_sentence_idx": len(sentences) - 1,
                "window_text": " ".join(sentences),
            }
        ]

    windows: List[Dict[str, Any]] = []
    idx = 0
    for i in range(0, len(sentences) - window_size + 1, stride):
        win_sents = sentences[i : i + window_size]
        windows.append(
            {
                "window_index": idx,
                "start_sentence_idx": i,
                "end_sentence_idx": i + window_size - 1,
                "window_text": " ".join(win_sents),
            }
        )
        idx += 1

    return windows


def extract_anchor_tokens(text: str) -> Set[str]:
    """Extract substantive lexical tokens."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = clean.split()
    return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS}


def filter_windows_by_anchor(
    windows: List[Dict[str, Any]],
    quote: Optional[str],
    claim_text: str,
    strategy: str = "quote_anchored",
) -> List[Dict[str, Any]]:
    """Filter windows to only those that contain the quote or key claim anchors."""
    if not windows:
        return []
    if strategy == "max":
        return windows

    if quote and len(quote.strip()) >= 10:
        norm_quote = re.sub(r"\s+", " ", quote.strip().lower())
        quote_words = [w for w in norm_quote.split() if len(w) >= 3 and w not in _STOPWORDS]
        matched_windows: List[Dict[str, Any]] = []

        for win in windows:
            norm_win = re.sub(r"\s+", " ", win["window_text"].lower())
            if norm_quote in norm_win:
                win["has_quote_anchor"] = True
                win["anchor_overlap_score"] = 1.0
                matched_windows.append(win)
                continue

            if quote_words:
                found_words = sum(1 for w in quote_words if w in norm_win)
                overlap = found_words / len(quote_words)
                if overlap >= 0.70:
                    win["has_quote_anchor"] = True
                    win["anchor_overlap_score"] = round(overlap, 3)
                    matched_windows.append(win)

        if matched_windows:
            return matched_windows

    claim_anchors = extract_anchor_tokens(claim_text)
    if not claim_anchors:
        return windows[:2]

    scored_windows = []
    for win in windows:
        win_anchors = extract_anchor_tokens(win["window_text"])
        overlap_count = len(claim_anchors.intersection(win_anchors))
        score = overlap_count / len(claim_anchors)
        win["has_quote_anchor"] = False
        win["anchor_overlap_score"] = round(score, 3)
        scored_windows.append((score, win))

    scored_windows.sort(key=lambda x: x[0], reverse=True)
    best_score = scored_windows[0][0]

    eligible = [
        w for s, w in scored_windows
        if s >= 0.40 or (best_score > 0 and s >= best_score - 0.15)
    ]
    return eligible if eligible else [scored_windows[0][1]]


def evaluate_sentence_windows(
    claim_text: str,
    chunk_text: str,
    original_chunk_verdict: str,
    backend: Any,
    quote: Optional[str] = None,
    fact_check_passed: bool = True,
    quote_valid: bool = True,
    mode: str = "off",
    window_size: int = 3,
    aggregation: str = "quote_anchored",
    tau_entail: float = 0.60,
    tau_contra: float = 0.50,
) -> SentenceWindowResult:
    """Run sentence-window verification and enforce anti-hallucination safety invariants."""
    effective_mode = mode.lower().strip()
    effective_size = window_size
    effective_agg = aggregation.lower().strip()
    t_entail = tau_entail
    t_contra = tau_contra

    if effective_mode == "off":
        return SentenceWindowResult(
            mode="off",
            original_chunk_verdict=original_chunk_verdict,
            window_verdict=original_chunk_verdict,
            final_verdict=original_chunk_verdict,
            verdict_changed=False,
            change_direction=None,
            aggregation_strategy=effective_agg,
            safety_override_applied=False,
        )

    sentences = split_chunk_into_sentences(chunk_text)
    raw_windows = create_sentence_windows(sentences, window_size=effective_size)
    eligible_windows = filter_windows_by_anchor(
        raw_windows,
        quote=quote,
        claim_text=claim_text,
        strategy=effective_agg,
    )

    details: List[SentenceWindowDetail] = []
    has_contradiction = False
    max_entail_prob = 0.0
    winning_detail: Optional[SentenceWindowDetail] = None

    for win in eligible_windows:
        w_text = win["window_text"]
        if hasattr(backend, "predict"):
            probs = backend.predict([(w_text, claim_text)])[0]
        else:
            probs = backend.predict_proba(premise=w_text, hypothesis=claim_text)
        ep = float(probs.get("entailment", 0.0))
        np_ = float(probs.get("neutral", 0.0))
        cp = float(probs.get("contradiction", 0.0))

        if cp >= t_contra:
            win_verdict = "contradiction"
            has_contradiction = True
        elif ep >= t_entail:
            win_verdict = "entailment"
        else:
            win_verdict = "neutral"

        d = SentenceWindowDetail(
            window_index=win["window_index"],
            start_sentence_idx=win["start_sentence_idx"],
            end_sentence_idx=win["end_sentence_idx"],
            window_text=w_text,
            has_quote_anchor=win.get("has_quote_anchor", False),
            anchor_overlap_score=win.get("anchor_overlap_score", 0.0),
            entailment_prob=round(ep, 4),
            neutral_prob=round(np_, 4),
            contradiction_prob=round(cp, 4),
            verdict=win_verdict,
        )
        details.append(d)

        if ep > max_entail_prob:
            max_entail_prob = ep
            winning_detail = d

    if has_contradiction:
        raw_window_verdict = "contradiction"
    elif max_entail_prob >= t_entail:
        raw_window_verdict = "entailment"
    else:
        raw_window_verdict = "neutral"

    safety_override = False
    safety_reason = None
    window_verdict = raw_window_verdict

    if not fact_check_passed and raw_window_verdict == "entailment":
        window_verdict = "contradiction"
        safety_override = True
        safety_reason = "Fact check failed: numbers/dates mismatch prevents window entailment override."
    elif not quote_valid and raw_window_verdict == "entailment":
        window_verdict = "neutral"
        safety_override = True
        safety_reason = "Quote validation failed: invalid quote prevents window entailment override."

    verdict_changed = (window_verdict != original_chunk_verdict)
    change_direction = f"{original_chunk_verdict}_to_{window_verdict}" if verdict_changed else None

    final_verdict = window_verdict if effective_mode == "enforce" else original_chunk_verdict

    return SentenceWindowResult(
        mode=effective_mode,
        original_chunk_verdict=original_chunk_verdict,
        window_verdict=window_verdict,
        final_verdict=final_verdict,
        verdict_changed=verdict_changed,
        change_direction=change_direction,
        aggregation_strategy=effective_agg,
        total_sentences=len(sentences),
        total_windows=len(raw_windows),
        evaluated_windows=len(eligible_windows),
        winning_window=winning_detail,
        all_window_details=details,
        safety_override_applied=safety_override,
        safety_override_reason=safety_reason,
    )

