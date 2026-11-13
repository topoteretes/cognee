"""Verbatim Quote Grounding Validator.

Validates that short verbatim quotes accompanying claims exist as exact or
whitespace/case-normalized substrings inside cited evidence chunks before NLI inference runs.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from examples.cookbooks.trustrag.core.claims import ExtractedClaim

logger = logging.getLogger(__name__)

_QUOTE_TRANS = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
    "–": "-",  # en-dash
    "—": "-",  # em-dash
    "…": "...",  # unicode ellipsis
    "\u00a0": " ",  # non-breaking space
    "\u200b": "",   # zero-width space
})


def normalize_text_for_quote_match(text: str) -> str:
    """Normalize text for robust substring quote matching."""
    if not text:
        return ""
    trans = text.translate(_QUOTE_TRANS)
    normalized = re.sub(r"\s+", " ", trans).strip()
    return normalized.lower()


def strip_outer_quotes(text: str) -> str:
    """Strip enclosing quotation marks if text is wrapped in quotes."""
    s = text.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    if (s.startswith("“") and s.endswith("”")) or (s.startswith("‘") and s.endswith("’")):
        return s[1:-1].strip()
    return s


class QuoteValidationResult(BaseModel):
    """Result of validating a claim's supporting quote against evidence chunk(s)."""

    passed: bool = Field(description="True if the quote was verified to exist in cited evidence")
    quote: Optional[str] = Field(default=None, description="The quote string checked")
    normalized_quote: Optional[str] = Field(default=None, description="Normalized quote checked")
    matching_chunk_id: Optional[str] = Field(
        default=None,
        description="ID of the cited chunk where the quote was successfully located",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Diagnostic narrative or failure reason",
    )
    match_method: Optional[str] = Field(
        default=None,
        description="Matching method that succeeded ('verbatim_exact', 'normalized_substring', 'elision_substring')",
    )


def validate_quote_in_text(
    quote: str,
    chunk_text: str,
    min_chars: int = 10,
    min_words: int = 2,
    min_segment_chars: int = 6,
    min_segment_words: int = 1,
) -> Tuple[bool, str, Optional[str]]:
    """Check if quote exists in chunk_text using exact, normalized, or linear-time elision matching."""
    if not quote or not quote.strip():
        return False, "empty_quote", None

    clean_quote = strip_outer_quotes(quote)
    if not clean_quote:
        return False, "empty_quote_after_strip", None

    words = clean_quote.split()
    if len(clean_quote) < min_chars or len(words) < min_words:
        return (
            False,
            f"quote_below_minimum_length: {len(clean_quote)} chars, {len(words)} words (min {min_chars} chars, {min_words} words)",
            None,
        )

    # 1. Exact verbatim match
    if clean_quote in chunk_text:
        return True, "verbatim_exact_match", "verbatim_exact"

    # 2. Whitespace and case normalized substring match
    norm_quote = normalize_text_for_quote_match(clean_quote)
    norm_chunk = normalize_text_for_quote_match(chunk_text)

    if norm_quote in norm_chunk:
        return True, "normalized_substring_match", "normalized_substring"

    # 3. Punctuation-tolerant match: strip trailing punctuation from quote
    stripped_punct_quote = re.sub(r"^[\s,.;:!?]+|[\s,.;:!?]+$", "", norm_quote)
    if stripped_punct_quote and stripped_punct_quote in norm_chunk:
        return True, "punctuation_trimmed_match", "normalized_substring"

    # 4. Strictly linear-time elision match if quote contains ellipsis ("..." or "…")
    if "..." in norm_quote:
        parts = [p.strip() for p in norm_quote.split("...") if p.strip()]
        if len(parts) >= 2:
            for part in parts:
                p_words = part.split()
                if len(part) < min_segment_chars or len(p_words) < min_segment_words:
                    return (
                        False,
                        f"elision_segment_too_short: '{part}' has {len(part)} chars (min {min_segment_chars})",
                        None,
                    )
            current_pos = 0
            all_found = True
            for part in parts:
                idx = norm_chunk.find(part, current_pos)
                if idx == -1:
                    all_found = False
                    break
                current_pos = idx + len(part)
            if all_found:
                return True, "elision_substring_match", "elision_substring"

    return False, "quote_not_found_in_chunk", None


def validate_claim_quote(
    claim: ExtractedClaim,
    chunk_texts: Dict[str, str],
    require_quote: bool = True,
    min_chars: int = 10,
    min_words: int = 2,
    min_segment_chars: int = 6,
    min_segment_words: int = 1,
) -> QuoteValidationResult:
    """Validate a claim's supporting quote against its cited chunk texts."""
    if not claim.cited_chunk_ids:
        return QuoteValidationResult(
            passed=False,
            quote=claim.quote,
            reason="claim_has_no_citations",
        )

    candidate_quotes: List[Tuple[Optional[str], str]] = []
    if claim.quote and claim.quote.strip():
        candidate_quotes.append((None, claim.quote.strip()))
    if claim.quotes:
        for cid, q in claim.quotes.items():
            if q and q.strip() and (None, q.strip()) not in candidate_quotes:
                candidate_quotes.append((cid, q.strip()))

    if not candidate_quotes:
        if require_quote:
            return QuoteValidationResult(
                passed=False,
                quote=None,
                reason="missing_verbatim_quote",
            )
        else:
            return QuoteValidationResult(
                passed=True,
                quote=None,
                reason="quote_omitted_not_required",
            )

    last_failure_reason = None
    for target_cid, quote_str in candidate_quotes:
        clean_q = strip_outer_quotes(quote_str)
        norm_q = normalize_text_for_quote_match(clean_q)

        cids_to_check = [target_cid] if (target_cid and target_cid in chunk_texts) else claim.cited_chunk_ids

        for cid in cids_to_check:
            text = chunk_texts.get(cid, "")
            if not text:
                continue

            passed, reason, method = validate_quote_in_text(
                clean_q,
                text,
                min_chars=min_chars,
                min_words=min_words,
                min_segment_chars=min_segment_chars,
                min_segment_words=min_segment_words,
            )
            if passed:
                return QuoteValidationResult(
                    passed=True,
                    quote=quote_str,
                    normalized_quote=norm_q,
                    matching_chunk_id=cid,
                    reason=reason,
                    match_method=method,
                )
            else:
                last_failure_reason = reason

    primary_quote = candidate_quotes[0][1]
    failure_msg = (
        last_failure_reason
        if last_failure_reason and (
            "quote_below_minimum_length" in last_failure_reason
            or "elision_segment_too_short" in last_failure_reason
        )
        else f"quote_not_found_in_evidence: '{primary_quote[:80]}'"
    )
    return QuoteValidationResult(
        passed=False,
        quote=primary_quote,
        normalized_quote=normalize_text_for_quote_match(strip_outer_quotes(primary_quote)),
        matching_chunk_id=None,
        reason=failure_msg,
        match_method=None,
    )

