"""Deterministic Fact Checking Engine for TrustRAG.

Provides high-precision deterministic factual consistency checks for numbers,
currencies, percentages, dates, and advisory checks for modals and negations.
Zero LLM latency and zero external dependencies beyond Pydantic.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field


class FactCheckType(str, Enum):
    """Category of factual anchor checked."""

    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    MODAL = "modal"
    NEGATION = "negation"


class FactCheckStatus(str, Enum):
    """Outcome status for an individual factual anchor check."""

    PASSED = "passed"
    MISMATCH = "mismatch"
    VALUE_NOT_FOUND = "value_not_found"
    NOT_CHECKABLE = "not_checkable"
    ADVISORY = "advisory"


class FactCheckDetail(BaseModel):
    """Detailed result for an individual fact check category."""

    check_type: FactCheckType = Field(description="Category of the fact check")
    status: FactCheckStatus = Field(description="Pass/mismatch/not_found/advisory status")
    claim_values: List[str] = Field(default_factory=list, description="Raw/extracted values in claim")
    chunk_values: List[str] = Field(default_factory=list, description="Extracted values in cited chunks")
    message: str = Field(description="Diagnostic explanation of check outcome")


class FactCheckResult(BaseModel):
    """Overall outcome across all deterministic checks for a single claim."""

    passed: bool = Field(description="True if all enforced checks passed or were not checkable")
    checks: List[FactCheckDetail] = Field(default_factory=list, description="Per-check detailed results")
    downgrade_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason for downgrading ENTAILMENT -> NEUTRAL",
    )


WORD_TO_NUMBER: Dict[str, float] = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty": 40.0,
    "fifty": 50.0,
    "sixty": 60.0,
    "seventy": 70.0,
    "eighty": 80.0,
    "ninety": 90.0,
    "hundred": 100.0,
    "thousand": 1000.0,
    "million": 1000000.0,
    "billion": 1000000000.0,
}

SCALE_MULTIPLIERS: Dict[str, float] = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
}

MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9, "sept": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_FORM_RE = re.compile(r"\b(?:Form|Schedule)\s+\d+(?:-[A-Za-z0-9]+)*\b", re.IGNORECASE)
_SECTION_RE = re.compile(
    r"\b(?:Section|Sec\.|Article|Clause|Paragraph|Rule|Notice|Item|Schedule|Exhibit)\s+[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*(?:\([A-Za-z0-9]+\))*",
    re.IGNORECASE,
)
_SECTION_JOIN_RE = re.compile(r"\b(?:or|and)\s+\d+(?:\([A-Za-z0-9]+\))+", re.IGNORECASE)
_STATUTE_RE = re.compile(
    r"\b(?:Securities\s+Act|Exchange\s+Act|Securities\s+Exchange\s+Act|Investment\s+Company\s+Act|Trust\s+Indenture\s+Act)\s+of\s+\d{4}\b",
    re.IGNORECASE,
)
_FILE_NO_RE = re.compile(
    r"\b(?:CIK|File\s+No\.?|Commission\s+File\s+Number)\s*[:#]?\s*\d+(?:-\d+)*\b",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"\b(?:Page|Pg\.?|P\.)\s+\d+\b", re.IGNORECASE)
_DOC_CHUNK_ID_RE = re.compile(r"\b[A-Za-z0-9_]+__[0-9]+__c[0-9]+\b")
_FOOTNOTE_RE = re.compile(r"\[\d+\]")
_ORDINAL_RE = re.compile(r"(?:^|[\n:;]|\s+)\d+[\.\)]\s+")


def strip_structural_citations(text: str) -> str:
    """Remove structural citations (sections, forms, statutes, pages, doc IDs, footnotes)."""
    cleaned = _FORM_RE.sub(" ", text)
    cleaned = _STATUTE_RE.sub(" ", cleaned)
    cleaned = _SECTION_RE.sub(" ", cleaned)
    cleaned = _SECTION_JOIN_RE.sub(" ", cleaned)
    cleaned = _FILE_NO_RE.sub(" ", cleaned)
    cleaned = _PAGE_RE.sub(" ", cleaned)
    cleaned = _DOC_CHUNK_ID_RE.sub(" ", cleaned)
    cleaned = _FOOTNOTE_RE.sub(" ", cleaned)
    cleaned = _ORDINAL_RE.sub(" ", cleaned)
    return cleaned


def parse_scaled_number(num_str: str, scale_str: Optional[str] = None) -> float:
    """Parse numeric string with optional commas and scale multiplier."""
    clean_num = num_str.replace(",", "").strip()
    val = float(clean_num)
    if scale_str:
        s_clean = scale_str.lower().strip()
        mult = SCALE_MULTIPLIERS.get(s_clean, 1.0)
        val *= mult
    return val


def extract_currencies(text: str) -> List[Tuple[str, float, str]]:
    """Extract currency values from text. Returns list of (code, canonical_amount, raw)."""
    results: List[Tuple[str, float, str]] = []

    pat1 = re.compile(
        r"(?:\$|USD\s*|US\$\s*|€|EUR\s*|£|GBP\s*)\s*([\d,]+(?:\.\d+)?)\s*([MmBbKk]|million|billion|thousand)?\b",
        re.IGNORECASE,
    )
    for m in pat1.finditer(text):
        raw = m.group(0).strip()
        num_part = m.group(1)
        scale_part = m.group(2)
        code = "USD"
        if "€" in raw or "EUR" in raw.upper():
            code = "EUR"
        elif "£" in raw or "GBP" in raw.upper():
            code = "GBP"
        try:
            val = parse_scaled_number(num_part, scale_part)
            results.append((code, val, raw))
        except (ValueError, TypeError):
            continue

    pat2 = re.compile(
        r"\b([\d,]+(?:\.\d+)?)\s*([MmBbKk]|million|billion|thousand)?\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP)\b",
        re.IGNORECASE,
    )
    for m in pat2.finditer(text):
        raw = m.group(0).strip()
        num_part = m.group(1)
        scale_part = m.group(2)
        code = "USD"
        raw_lower = raw.lower()
        if "euro" in raw_lower or "eur" in raw_lower:
            code = "EUR"
        elif "pound" in raw_lower or "gbp" in raw_lower:
            code = "GBP"
        try:
            val = parse_scaled_number(num_part, scale_part)
            if not any(r[2] == raw or abs(r[1] - val) < 1e-4 for r in results):
                results.append((code, val, raw))
        except (ValueError, TypeError):
            continue

    return results


def extract_percentages(text: str) -> List[Tuple[float, str]]:
    """Extract percentage values from text. Returns list of (canonical_percentage, raw)."""
    results: List[Tuple[float, str]] = []

    pat = re.compile(
        r"\b([\d,]+(?:\.\d+)?)\s*(?:%|percent\b|percentage\s+points?\b)",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        raw = m.group(0).strip()
        try:
            val = float(m.group(1).replace(",", ""))
            results.append((val, raw))
        except (ValueError, TypeError):
            continue

    for word, num in WORD_TO_NUMBER.items():
        if word in ("hundred", "thousand", "million", "billion"):
            continue
        w_pat = re.compile(rf"\b{word}\s+percent\b", re.IGNORECASE)
        for m in w_pat.finditer(text):
            results.append((num, m.group(0).strip()))

    return results


DateTuple = Tuple[Optional[int], Optional[int], Optional[int]]


def extract_dates(text: str) -> List[Tuple[DateTuple, str]]:
    """Extract dates from text in common corporate/legal formats."""
    results: List[Tuple[DateTuple, str]] = []
    seen_raw: Set[str] = set()

    iso_pat = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
    for m in iso_pat.finditer(text):
        raw = m.group(0).strip()
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            results.append(((y, mo, d), raw))
            seen_raw.add(raw)

    month_names = "|".join(sorted(MONTH_MAP.keys(), key=len, reverse=True))

    named_pat1 = re.compile(
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?",
        re.IGNORECASE,
    )
    for m in named_pat1.finditer(text):
        raw = m.group(0).strip()
        if raw in seen_raw:
            continue
        mo_str = m.group(1).lower()
        mo = MONTH_MAP.get(mo_str)
        d = int(m.group(2))
        y = int(m.group(3)) if m.group(3) else None
        if mo and 1 <= d <= 31:
            results.append(((y, mo, d), raw))
            seen_raw.add(raw)

    named_pat2 = re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})(?:,?\s+(\d{{4}}))?",
        re.IGNORECASE,
    )
    for m in named_pat2.finditer(text):
        raw = m.group(0).strip()
        if raw in seen_raw:
            continue
        d = int(m.group(1))
        mo_str = m.group(2).lower()
        mo = MONTH_MAP.get(mo_str)
        y = int(m.group(3)) if m.group(3) else None
        if mo and 1 <= d <= 31:
            results.append(((y, mo, d), raw))
            seen_raw.add(raw)

    named_pat3 = re.compile(
        rf"\b({month_names})\s+(\d{{4}})\b",
        re.IGNORECASE,
    )
    for m in named_pat3.finditer(text):
        raw = m.group(0).strip()
        if raw in seen_raw:
            continue
        mo_str = m.group(1).lower()
        mo = MONTH_MAP.get(mo_str)
        y = int(m.group(2))
        if mo and 1900 <= y <= 2100:
            results.append(((y, mo, None), raw))
            seen_raw.add(raw)

    return results


def date_matches(claim_date: DateTuple, chunk_date: DateTuple) -> bool:
    """Check if claim date is consistent with chunk date, supporting partial dates."""
    cy, cmo, cd = claim_date
    ky, kmo, kd = chunk_date

    if cmo is not None and kmo is not None and cmo != kmo:
        return False
    if cd is not None and kd is not None and cd != kd:
        return False
    if cy is not None and ky is not None and cy != ky:
        return False
    return True


def extract_numbers(
    text: str,
    exclude_currency_spans: Optional[Sequence[str]] = None,
    exclude_percentage_spans: Optional[Sequence[str]] = None,
    exclude_date_spans: Optional[Sequence[str]] = None,
) -> List[Tuple[float, str]]:
    """Extract standalone numbers from text, excluding currencies, percentages, and dates."""
    cleaned = strip_structural_citations(text)
    to_mask = list(exclude_currency_spans or []) + list(exclude_percentage_spans or []) + list(exclude_date_spans or [])
    for span in to_mask:
        cleaned = cleaned.replace(span, " ")

    results: List[Tuple[float, str]] = []

    pat = re.compile(r"\b([\d,]+(?:\.\d+)?)\b")
    for m in pat.finditer(cleaned):
        raw = m.group(0).strip()
        try:
            val = float(raw.replace(",", ""))
            results.append((val, raw))
        except (ValueError, TypeError):
            continue

    for word, val in WORD_TO_NUMBER.items():
        if word in ("hundred", "thousand", "million", "billion", "zero"):
            continue
        w_pat = re.compile(rf"\b{word}\b", re.IGNORECASE)
        for m in w_pat.finditer(cleaned):
            results.append((val, m.group(0).strip()))

    return results


def extract_modals(text: str) -> List[Tuple[str, str]]:
    """Extract modal verbs for advisory checks."""
    results: List[Tuple[str, str]] = []
    obligatory_pat = re.compile(r"\b(shall|must|required\s+to|obligated\s+to)\b", re.IGNORECASE)
    permissive_pat = re.compile(r"\b(may|can|might|permitted\s+to)\b", re.IGNORECASE)

    for m in obligatory_pat.finditer(text):
        results.append(("obligatory", m.group(0).strip()))
    for m in permissive_pat.finditer(text):
        results.append(("permissive", m.group(0).strip()))

    return results


def extract_negations(text: str) -> List[str]:
    """Extract negation markers for advisory checks."""
    neg_pat = re.compile(
        r"\b(not|no|never|neither|nor|cannot|can't|won't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b",
        re.IGNORECASE,
    )
    return [m.group(0).strip() for m in neg_pat.finditer(text)]


def detect_table_scaling(chunk_text: str) -> float:
    """Detect if evidence contains financial header scaling like '(in thousands)'."""
    lower = chunk_text.lower()
    if "in thousands" in lower or "(in thousands)" in lower:
        return 1_000.0
    if "in millions" in lower or "(in millions)" in lower:
        return 1_000_000.0
    if "in billions" in lower or "(in billions)" in lower:
        return 1_000_000_000.0
    return 1.0


def check_claim_against_chunks(
    claim_text: str,
    chunk_texts: Sequence[str],
) -> FactCheckResult:
    """Run all deterministic fact checks on a claim against cited chunks."""
    if not chunk_texts:
        return FactCheckResult(
            passed=True,
            checks=[
                FactCheckDetail(
                    check_type=FactCheckType.NUMBER,
                    status=FactCheckStatus.NOT_CHECKABLE,
                    claim_values=[],
                    chunk_values=[],
                    message="No cited chunk texts supplied for fact checking.",
                )
            ],
            downgrade_reason=None,
        )

    combined_chunk_text = "\n\n".join(chunk_texts)
    table_scale = detect_table_scaling(combined_chunk_text)

    checks: List[FactCheckDetail] = []
    failures: List[str] = []

    # 1. Currency Checks
    claim_curr = extract_currencies(claim_text)
    chunk_curr = extract_currencies(combined_chunk_text)

    if claim_curr:
        chunk_vals_repr = [f"{c[0]} {c[1]:,.2f} ({c[2]})" for c in chunk_curr]
        candidate_amounts: List[Tuple[str, float]] = [(c[0], c[1]) for c in chunk_curr]
        raw_chunk_nums = extract_numbers(combined_chunk_text)
        if table_scale > 1.0 or chunk_curr:
            for ch_num, _ in raw_chunk_nums:
                candidate_amounts.append(("USD", ch_num * table_scale))
                if table_scale > 1.0:
                    candidate_amounts.append(("USD", ch_num))

        for code, amount, raw in claim_curr:
            matched = False
            for ch_code, ch_amount in candidate_amounts:
                diff = abs(amount - ch_amount)
                max_val = max(amount, ch_amount)
                if diff < 1.0 or (max_val > 0 and (diff / max_val) < 0.005):
                    matched = True
                    break

            if matched:
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.CURRENCY,
                        status=FactCheckStatus.PASSED,
                        claim_values=[raw],
                        chunk_values=chunk_vals_repr or [f"USD {amount:,.0f}"],
                        message=f"Currency '{raw}' verified in cited evidence.",
                    )
                )
            elif candidate_amounts:
                msg = f"Currency mismatch: claim specifies '{raw}' ({amount:,.0f}), but cited evidence contains {chunk_vals_repr or [a[1] for a in candidate_amounts[:5]]}"
                failures.append(msg)
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.CURRENCY,
                        status=FactCheckStatus.MISMATCH,
                        claim_values=[raw],
                        chunk_values=chunk_vals_repr,
                        message=msg,
                    )
                )
            else:
                msg = f"Currency not found: claim specifies '{raw}' ({amount:,.0f}), but cited evidence contains no currency values."
                failures.append(msg)
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.CURRENCY,
                        status=FactCheckStatus.VALUE_NOT_FOUND,
                        claim_values=[raw],
                        chunk_values=[],
                        message=msg,
                    )
                )

    # 2. Percentage Checks
    claim_pct = extract_percentages(claim_text)
    chunk_pct = extract_percentages(combined_chunk_text)

    if claim_pct:
        chunk_vals_repr = [f"{p[0]}% ({p[1]})" for p in chunk_pct]
        for val, raw in claim_pct:
            matched = any(abs(val - ch_val) < 1e-3 for ch_val, _ in chunk_pct)
            if matched:
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.PERCENTAGE,
                        status=FactCheckStatus.PASSED,
                        claim_values=[raw],
                        chunk_values=chunk_vals_repr,
                        message=f"Percentage '{raw}' ({val}%) verified in cited evidence.",
                    )
                )
            elif chunk_pct:
                msg = f"Percentage mismatch: claim specifies '{raw}' ({val}%), but cited evidence contains {chunk_vals_repr}"
                failures.append(msg)
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.PERCENTAGE,
                        status=FactCheckStatus.MISMATCH,
                        claim_values=[raw],
                        chunk_values=chunk_vals_repr,
                        message=msg,
                    )
                )
            else:
                msg = f"Percentage not found: claim specifies '{raw}' ({val}%), but cited evidence contains no percentage values."
                failures.append(msg)
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.PERCENTAGE,
                        status=FactCheckStatus.VALUE_NOT_FOUND,
                        claim_values=[raw],
                        chunk_values=[],
                        message=msg,
                    )
                )

    # 3. Date Checks
    claim_dates = extract_dates(claim_text)
    chunk_dates = extract_dates(combined_chunk_text)

    if claim_dates:
        chunk_vals_repr = [d[1] for d in chunk_dates]
        for dt_tuple, raw in claim_dates:
            matched = any(date_matches(dt_tuple, ch_dt) for ch_dt, _ in chunk_dates)
            if matched:
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.DATE,
                        status=FactCheckStatus.PASSED,
                        claim_values=[raw],
                        chunk_values=chunk_vals_repr,
                        message=f"Date '{raw}' verified in cited evidence.",
                    )
                )
            elif chunk_dates:
                msg = f"Date mismatch: claim specifies '{raw}', but cited evidence contains {chunk_vals_repr}"
                failures.append(msg)
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.DATE,
                        status=FactCheckStatus.MISMATCH,
                        claim_values=[raw],
                        chunk_values=chunk_vals_repr,
                        message=msg,
                    )
                )
            else:
                msg = f"Date not found: claim specifies '{raw}', but cited evidence contains no date values."
                failures.append(msg)
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.DATE,
                        status=FactCheckStatus.VALUE_NOT_FOUND,
                        claim_values=[raw],
                        chunk_values=[],
                        message=msg,
                    )
                )

    # 4. Standalone Number Checks
    curr_spans = [c[2] for c in claim_curr]
    pct_spans = [p[1] for p in claim_pct]
    date_spans = [d[1] for d in claim_dates]
    claim_nums = extract_numbers(
        claim_text,
        exclude_currency_spans=curr_spans,
        exclude_percentage_spans=pct_spans,
        exclude_date_spans=date_spans,
    )

    ch_curr_spans = [c[2] for c in chunk_curr]
    ch_pct_spans = [p[1] for p in chunk_pct]
    ch_date_spans = [d[1] for d in chunk_dates]
    chunk_nums = extract_numbers(
        combined_chunk_text,
        exclude_currency_spans=ch_curr_spans,
        exclude_percentage_spans=ch_pct_spans,
        exclude_date_spans=ch_date_spans,
    )

    if claim_nums:
        chunk_vals_repr = [f"{n[0]} ({n[1]})" for n in chunk_nums]
        for val, raw in claim_nums:
            matched = any(abs(val - ch_val) < 1e-4 for ch_val, _ in chunk_nums)
            if matched:
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.NUMBER,
                        status=FactCheckStatus.PASSED,
                        claim_values=[raw],
                        chunk_values=chunk_vals_repr,
                        message=f"Number '{raw}' ({val}) verified in cited evidence.",
                    )
                )
            elif chunk_nums:
                msg = f"Number mismatch: claim specifies '{raw}' ({val}), but cited evidence contains {[n[0] for n in chunk_nums[:5]]}"
                failures.append(msg)
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.NUMBER,
                        status=FactCheckStatus.MISMATCH,
                        claim_values=[raw],
                        chunk_values=chunk_vals_repr,
                        message=msg,
                    )
                )
            else:
                msg = f"Number not found: claim specifies '{raw}' ({val}), but cited evidence contains no matching numeric values."
                failures.append(msg)
                checks.append(
                    FactCheckDetail(
                        check_type=FactCheckType.NUMBER,
                        status=FactCheckStatus.VALUE_NOT_FOUND,
                        claim_values=[raw],
                        chunk_values=[],
                        message=msg,
                    )
                )

    # 5. Advisory Checks (Modals & Negation)
    claim_modals = extract_modals(claim_text)
    chunk_modals = extract_modals(combined_chunk_text)
    if claim_modals:
        checks.append(
            FactCheckDetail(
                check_type=FactCheckType.MODAL,
                status=FactCheckStatus.ADVISORY,
                claim_values=[f"{m[0]}:{m[1]}" for m in claim_modals],
                chunk_values=[f"{m[0]}:{m[1]}" for m in chunk_modals],
                message=f"Modal verb observation: claim uses {[m[1] for m in claim_modals]}.",
            )
        )

    claim_negs = extract_negations(claim_text)
    chunk_negs = extract_negations(combined_chunk_text)
    if claim_negs:
        checks.append(
            FactCheckDetail(
                check_type=FactCheckType.NEGATION,
                status=FactCheckStatus.ADVISORY,
                claim_values=claim_negs,
                chunk_values=chunk_negs,
                message=f"Negation marker observation: claim uses {claim_negs}.",
            )
        )

    if not failures:
        if not checks:
            checks.append(
                FactCheckDetail(
                    check_type=FactCheckType.NUMBER,
                    status=FactCheckStatus.NOT_CHECKABLE,
                    claim_values=[],
                    chunk_values=[],
                    message="Claim contains no checkable numerical, currency, percentage, or date anchors.",
                )
            )
        return FactCheckResult(passed=True, checks=checks, downgrade_reason=None)

    return FactCheckResult(passed=False, checks=checks, downgrade_reason="; ".join(failures))

