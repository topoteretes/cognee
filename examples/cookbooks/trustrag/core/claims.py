"""Atomic Claim Extraction with Citation Provenance Preservation.

Decomposes a draft answer into atomic, individually checkable claims
while strictly preserving inline bracketed citation provenance ([chunk_id]).
Provides both rule-based deterministic extraction and LLM-based extraction.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(
    r"\[\s*([A-Za-z0-9_-]+)(?:\s*(?:\||:|,)\s*(?:quote\s*:\s*)?(?:[\"“'‘](?:[^\"”'’]+)[\"”'’]|(?:[^\]]+)))?\s*\]"
)

_CITATION_WITH_QUOTE_RE = re.compile(
    r"\[\s*([A-Za-z0-9_-]+)(?:\s*(?:\||:|,)\s*(?:quote\s*:\s*)?(?:[\"“'‘]([^\"”'’]+)[\"”'’]|([^\]]+)))?\s*\]"
)

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
    re.compile(
        r"there\s+is\s+no\s+(?:information|mention|evidence|data)\s+(?:regarding|about|concerning)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"no\s+information\s+(?:is\s+available|regarding|about)\b",
        re.IGNORECASE,
    ),
]

_META_CLAIM_PATTERNS = [
    re.compile(
        r"^(?:the\s+(?:available\s+|provided\s+|supplied\s+)?evidence|"
        r"the\s+retrieved\s+(?:chunks?|documents?|passages?)|"
        r"the\s+(?:provided\s+)?text|"
        r"the\s+cited\s+chunk)\s+"
        r"(?:mentions?|states?|discusses?|describes?|indicates?|notes?|defines?|contains?|refers?\s+to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:according\s+to|based\s+on)\s+the\s+(?:provided\s+|retrieved\s+|available\s+)?"
        r"(?:evidence|chunks?|documents?|text)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^another\s+chunk\s+(?:refers?|mentions?|states?)\b",
        re.IGNORECASE,
    ),
]

_AUX_AND_MODAL_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "do", "does", "did", "done",
    "shall", "will", "must", "may", "can", "could", "should", "would", "might",
}

_COMMON_VERB_ROOTS = {
    "mean", "means", "meaning", "meant",
    "include", "includes", "including", "included",
    "require", "requires", "requiring", "required",
    "provide", "provides", "providing", "provided",
    "operate", "operates", "operating", "operated",
    "govern", "governs", "governing", "governed",
    "amend", "amends", "amending", "amended",
    "specify", "specifies", "specifying", "specified",
    "define", "defines", "defining", "defined",
    "state", "states", "stating", "stated",
    "agree", "agrees", "agreeing", "agreed",
    "hold", "holds", "holding", "held",
    "warrant", "warrants", "warranting", "warranted",
    "terminate", "terminates", "terminating", "terminated",
    "apply", "applies", "applying", "applied",
    "relate", "relates", "relating", "related",
    "discuss", "discusses", "discussing", "discussed",
    "supervise", "supervises", "supervising", "supervised",
    "maintain", "maintains", "maintaining", "maintained",
    "establish", "establishes", "establishing", "established",
    "elect", "elects", "electing", "elected",
    "approve", "approves", "approving", "approved",
    "ratify", "ratifies", "ratifying", "ratified",
    "contain", "contains", "containing", "contained",
    "pay", "pays", "paying", "paid",
    "charge", "charges", "charging", "charged",
    "cost", "costs", "costing",
    "cover", "covers", "covering", "covered",
    "prohibit", "prohibits", "prohibiting", "prohibited",
    "allow", "allows", "allowing", "allowed",
    "permit", "permits", "permitting", "permitted",
    "expire", "expires", "expiring", "expired",
    "commence", "commences", "commencing", "commenced",
    "consist", "consists", "consisting", "consisted",
    "adopt", "adopts", "adopting", "adopted",
    "announce", "announces", "announcing", "announced",
    "identify", "identifies", "identifying", "identified",
    "handle", "handles", "handling", "handled",
    "review", "reviews", "reviewing", "reviewed",
    "cease", "ceases", "ceasing", "ceased",
    "found", "founds", "founding", "founded",
    "file", "files", "filing", "filed",
    "occur", "occurs", "occurring", "occurred",
    "reflect", "reflects", "reflecting", "reflected",
    "continue", "continues", "continuing", "continued",
    "receive", "receives", "receiving", "received",
    "rest", "rests", "resting", "rested",
    "accrue", "accrues", "accruing", "accrued",
    "supply", "supplies", "supplying", "supplied",
    "headquarter", "headquarters", "headquartering", "headquartered",
    "reference", "references", "referencing", "referenced",
    "refer", "refers", "referring", "referred",
    "disclose", "discloses", "disclosing", "disclosed",
    "indicate", "indicates", "indicating", "indicated",
    "show", "shows", "showing", "showed", "shown",
    "report", "reports", "reporting", "reported",
    "describe", "describes", "describing", "described",
    "note", "notes", "noting", "noted",
    "list", "lists", "listing", "listed",
    "set", "sets", "setting",
    "stipulate", "stipulates", "stipulating", "stipulated",
    "outline", "outlines", "outlining", "outlined",
    "detail", "details", "detailing", "detailed",
    "exclude", "excludes", "excluding", "excluded",
    "bear", "bears", "bearing", "borne",
    "owe", "owes", "owing", "owed",
    "own", "owns", "owning", "owned",
    "designate", "designates", "designating", "designated",
    "appoint", "appoints", "appointing", "appointed",
    "serve", "serves", "serving", "served",
    "earn", "earns", "earning", "earned",
    "generate", "generates", "generating", "generated",
    "incur", "incurs", "incurring", "incurred",
    "exceed", "exceeds", "exceeding", "exceeded",
    "represent", "represents", "representing", "represented",
    "constitute", "constitutes", "constituting", "constituted",
    "comply", "complies", "complying", "complied",
    "conduct", "conducts", "conducting", "conducted",
    "perform", "performs", "performing", "performed",
    "execute", "executes", "executing", "executed",
    "deliver", "delivers", "delivering", "delivered",
    "transfer", "transfers", "transferring", "transferred",
    "assign", "assigns", "assigning", "assigned",
    "vest", "vests", "vesting", "vested",
    "grant", "grants", "granting", "granted",
    "exercise", "exercises", "exercising", "exercised",
    "settle", "settles", "settling", "settled",
    "issue", "issues", "issuing", "issued",
    "purchase", "purchases", "purchasing", "purchased",
    "sell", "sells", "selling", "sold",
    "buy", "buys", "buying", "bought",
    "invest", "invests", "investing", "invested",
    "allocate", "allocates", "allocating", "allocated",
    "distribute", "distributes", "distributing", "distributed",
    "reimburse", "reimburses", "reimbursing", "reimbursed",
    "compensate", "compensates", "compensating", "compensated",
    "repay", "repays", "repaying", "repaid",
    "borrow", "borrows", "borrowing", "borrowed",
    "lend", "lends", "lending", "lent",
    "fund", "funds", "funding", "funded",
    "finance", "finances", "financing", "financed",
    "manage", "manages", "managing", "managed",
    "direct", "directs", "directing", "directed",
    "lead", "leads", "leading", "led",
    "employ", "employs", "employing", "employed",
    "hire", "hires", "hiring", "hired",
    "cancel", "cancels", "canceling", "cancelled", "canceled",
    "renew", "renews", "renewing", "renewed",
    "extend", "extends", "extending", "extended",
    "suspend", "suspends", "suspending", "suspended",
    "resume", "resumes", "resuming", "resumed",
    "satisfy", "satisfies", "satisfying", "satisfied",
    "meet", "meets", "meeting", "met",
    "fail", "fails", "failing", "failed",
    "breach", "breaches", "breaching", "breached",
    "default", "defaults", "defaulting", "defaulted",
    "violate", "violates", "violating", "violated",
    "remedy", "remedies", "remedying", "remedied",
    "cure", "cures", "curing", "cured",
    "waive", "waives", "waiving", "waived",
    "indemnify", "indemnifies", "indemnifying", "indemnified",
    "protect", "protects", "protecting", "protected",
    "insure", "insures", "insuring", "insured",
    "secure", "secures", "securing", "secured",
    "guarantee", "guarantees", "guaranteeing", "guaranteed",
}

DEFAULT_CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are a precise factual claim extraction assistant for financial and legal documents.
Your task is to decompose the provided draft answer into atomic, self-contained, individually-checkable factual claims.

STRICT EXTRACTION RULES:
1. ATOMIC PROPOSITIONS: Each claim must represent a single, independent factual proposition. Do NOT combine multiple facts into one claim.
2. SELF-CONTAINED: Resolve ambiguous pronouns (such as "it", "they", "this company", "the agreement") to explicit referents using context from the query and answer.
3. CITATION PRESERVATION: Preserve all inline bracketed citations (e.g. `[chunk_id]`) from the original answer.
   - Attach every chunk ID that supports a claim to that claim's `cited_chunks` list.
   - ONLY use chunk IDs that literally appear in the input answer.
   - NEVER invent, modify, normalize, or infer chunk IDs.
   - If a statement has no inline citation in the input, set `cited_chunks: []`.
   - If a citation contains a verbatim quote (e.g. `[chunk_id | quote: "..."]`), include it in the `quote` field.
4. NO EXTRAPOLATION: Do not introduce facts, numbers, or dates that are not stated in the draft answer.
5. REFUSALS: If the text conveys that evidence is insufficient or unavailable, do not invent facts.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects with the following schema:
[
  {
    "claim": "Atomic factual proposition with pronouns resolved.",
    "cited_chunks": ["chunk_id_1"],
    "quote": "Short verbatim supporting quote from chunk if present, else null.",
    "source_sentence": "The original sentence containing this claim."
  }
]
"""


class ExtractedClaim(BaseModel):
    """An atomic proposition with preserved citation attribution."""

    claim_id: str = Field(description="Deterministic claim identifier (e.g. 'claim_001')")
    claim_text: str = Field(description="Atomic proposition text with pronouns resolved and citations removed")
    cited_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Exact chunk IDs cited in support of this proposition",
    )
    source_sentence: str = Field(
        description="The original sentence from which this claim was decomposed"
    )
    is_cited: bool = Field(
        default=False,
        description="True if cited_chunk_ids is non-empty, False for uncited assertions",
    )
    quote: Optional[str] = Field(
        default=None,
        description="Verbatim supporting quote from cited evidence chunk",
    )
    quotes: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of cited chunk_id -> verbatim supporting quote",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_is_cited(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cits = data.get("cited_chunk_ids", [])
            data["is_cited"] = bool(cits and len(cits) > 0)
        return data


class ClaimExtractionResult(BaseModel):
    """Result container for decomposed claims and extraction provenance."""

    query: Optional[str] = Field(default=None, description="The original user question, if supplied")
    draft_answer: str = Field(description="The original answer text that was decomposed")
    claims: List[ExtractedClaim] = Field(default_factory=list, description="List of atomic claims")
    total_claims: int = Field(default=0, description="Total number of extracted claims")
    cited_claims_count: int = Field(default=0, description="Number of claims with at least one citation")
    uncited_claims_count: int = Field(default=0, description="Number of uncited claims")
    extraction_method: str = Field(description="Method used: 'llm', 'rule_based', or 'mock'")
    provider: Optional[str] = Field(default=None, description="LLM provider name, if LLM was used")
    model: Optional[str] = Field(default=None, description="LLM model name, if LLM was used")

    @model_validator(mode="before")
    @classmethod
    def populate_counts(cls, data: Any) -> Any:
        if isinstance(data, dict) and "claims" in data:
            claims = data["claims"]
            data.setdefault("total_claims", len(claims))
            cited = sum(
                1
                for c in claims
                if (c.is_cited if hasattr(c, "is_cited") else (isinstance(c, dict) and c.get("is_cited", False)))
            )
            data.setdefault("cited_claims_count", cited)
            data.setdefault("uncited_claims_count", len(claims) - cited)
        return data


def extract_citations_from_text(text: str) -> List[str]:
    """Extract all [chunk_id] inline citations from text, preserving order of first appearance."""
    matches = _CITATION_PATTERN.findall(text)
    seen: Set[str] = set()
    result: List[str] = []
    for cid in matches:
        if cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


def extract_citations_and_quotes_from_text(text: str) -> List[Tuple[str, Optional[str]]]:
    """Extract all (chunk_id, quote) pairs from inline citations in text."""
    matches = _CITATION_WITH_QUOTE_RE.findall(text)
    result: List[Tuple[str, Optional[str]]] = []
    for m in matches:
        cid = m[0].strip()
        q = m[1].strip() if m[1] else (m[2].strip() if m[2] else None)
        result.append((cid, q))
    return result


def _clean_citations(text: str) -> str:
    """Remove bracketed citations and normalize whitespace and punctuation."""
    cleaned = _CITATION_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_refusal_sentence(sentence: str) -> bool:
    """Check if a sentence represents a refusal or statement of evidence insufficiency."""
    if not sentence or not sentence.strip():
        return False
    return any(p.search(sentence) is not None for p in _REFUSAL_PATTERNS)


def _is_meta_claim(text: str) -> bool:
    """Check if text is a meta-claim about the evidence documents."""
    clean = _clean_citations(text).strip()
    return any(p.search(clean) is not None for p in _META_CLAIM_PATTERNS)


def _has_predicate(text: str) -> bool:
    """Check if clean text contains at least one recognized verb/predicate."""
    clean = _clean_citations(text).strip().rstrip(".!?")
    tokens = re.findall(r"\b[A-Za-z]+(?:'[a-z]+)?\b", clean.lower())
    if len(tokens) < 2:
        return False
    for tok in tokens:
        if tok in _AUX_AND_MODAL_VERBS or tok in _COMMON_VERB_ROOTS:
            return True
        if tok.endswith("s") and len(tok) >= 4 and tok[:-1] in _COMMON_VERB_ROOTS:
            return True
        if tok.endswith("es") and len(tok) >= 5 and tok[:-2] in _COMMON_VERB_ROOTS:
            return True
        if tok.endswith("ed") and len(tok) >= 5:
            return True
        if tok.endswith("ing") and len(tok) >= 6:
            return True
    return False


def _split_into_sentences(text: str) -> List[str]:
    """Segment answer into sentences."""
    cleaned = text.strip()
    if not cleaned:
        return []
    raw_segments = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])|\n\n+", cleaned)
    return [s.strip() for s in raw_segments if s.strip()]


def _split_into_sentences_bracket_aware(text: str) -> List[str]:
    """Segment answer into sentences while preserving bracketed citations intact."""
    cleaned = text.strip()
    if not cleaned:
        return []
    sentences: List[str] = []
    current: List[str] = []
    in_bracket = 0
    i = 0
    n = len(cleaned)
    while i < n:
        ch = cleaned[i]
        if ch == '[':
            in_bracket += 1
        elif ch == ']' and in_bracket > 0:
            in_bracket -= 1
        current.append(ch)
        if in_bracket == 0:
            if ch in '.!?':
                j = i + 1
                while j < n and cleaned[j].isspace():
                    j += 1
                if j > i + 1 and j < n and (cleaned[j].isupper() or cleaned[j].isdigit() or cleaned[j] in '"\'['):
                    seg = "".join(current).strip()
                    if not re.search(r"\b(?:e\.g|i\.e|u\.s|mr|mrs|ms|dr|vs|sec)\.$", seg, re.IGNORECASE):
                        sentences.append(seg)
                        current = []
                        i = j - 1
            elif ch == '\n' and i + 1 < n and cleaned[i + 1] == '\n':
                seg = "".join(current).strip()
                if seg:
                    sentences.append(seg)
                    current = []
        i += 1
    last_seg = "".join(current).strip()
    if last_seg:
        sentences.append(last_seg)
    return sentences


def _split_outside_enclosures(text: str, pattern: str) -> List[str]:
    """Split text on regex pattern only when not inside parentheses, brackets, or quotes."""
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return [text]

    valid_splits = []
    for m in matches:
        prefix = text[:m.start()]
        p_depth = 0
        b_depth = 0
        in_quote = False
        for ch in prefix:
            if ch == '"':
                in_quote = not in_quote
            elif not in_quote:
                if ch == '(':
                    p_depth += 1
                elif ch == ')':
                    p_depth = max(0, p_depth - 1)
                elif ch == '[':
                    b_depth += 1
                elif ch == ']':
                    b_depth = max(0, b_depth - 1)
        if p_depth == 0 and b_depth == 0 and not in_quote:
            valid_splits.append((m.start(), m.end()))

    if not valid_splits:
        return [text]

    parts = []
    last_end = 0
    for start, end in valid_splits:
        part = text[last_end:start].strip()
        if part:
            parts.append(part)
        last_end = end
    tail = text[last_end:].strip()
    if tail:
        parts.append(tail)
    return parts


def _decompose_sentence(
    sentence: str,
    cleanup: bool = False,
) -> List[Tuple[str, List[str], Optional[str], Dict[str, str]]]:
    """Decompose a single sentence into atomic propositions."""
    sent_cits_quotes = extract_citations_and_quotes_from_text(sentence)
    sent_cits = [c[0] for c in sent_cits_quotes]
    sent_quotes_map = {c[0]: c[1] for c in sent_cits_quotes if c[1]}

    if "; " in sentence:
        if cleanup:
            candidate = _split_outside_enclosures(sentence, r";\s+")
        else:
            candidate = [c.strip() for c in sentence.split("; ") if c.strip()]
        if cleanup and not all(_has_predicate(c) for c in candidate):
            raw_clauses = [sentence]
        else:
            raw_clauses = candidate
    elif re.search(r",\s+(?:and|while|but)\s+", sentence):
        if cleanup:
            candidate = _split_outside_enclosures(sentence, r",\s+(?:and|while|but)\s+")
        else:
            candidate = [c.strip() for c in re.split(r",\s+(?:and|while|but)\s+", sentence) if c.strip()]
        if cleanup:
            if all(_has_predicate(c) for c in candidate) and not any(
                re.match(r"^(?:as|because|since|if|unless|while)\b", _clean_citations(c).strip(), re.I)
                for c in candidate[1:]
            ):
                raw_clauses = candidate
            else:
                raw_clauses = [sentence]
        else:
            raw_clauses = candidate
    elif re.search(
        r"\s+and\s+(?=(?:it|they|the\s+\w+|each|all|both|is|are|was|were|has|have|had|shall|will|must|may|can|provides|operates|requires|holds|states|warrants|agrees|includes|governs|specifies)\b)",
        sentence,
        re.IGNORECASE,
    ):
        pattern_and = r"\s+and\s+(?=(?:it|they|the\s+\w+|each|all|both|is|are|was|were|has|have|had|shall|will|must|may|can|provides|operates|requires|holds|states|warrants|agrees|includes|governs|specifies)\b)"
        if cleanup:
            candidate = _split_outside_enclosures(sentence, pattern_and)
        else:
            candidate = [
                c.strip()
                for c in re.split(pattern_and, sentence, flags=re.IGNORECASE)
                if c.strip()
            ]
        if cleanup and not all(_has_predicate(c) for c in candidate):
            raw_clauses = [sentence]
        else:
            raw_clauses = candidate
    else:
        raw_clauses = [sentence]

    subject = None
    if len(raw_clauses) > 1:
        subject_match = re.match(
            r"^(The\s+[A-Za-z0-9_-]+|Each\s+[A-Za-z0-9_-]+|[A-Z][a-z0-9_-]+(?:\s+[A-Z][a-z0-9_-]+)?)",
            raw_clauses[0],
        )
        if subject_match:
            subject = subject_match.group(1)

    results: List[Tuple[str, List[str], Optional[str], Dict[str, str]]] = []
    for idx, clause in enumerate(raw_clauses):
        clause_cits_quotes = extract_citations_and_quotes_from_text(clause)
        clause_cits = [c[0] for c in clause_cits_quotes]
        clause_quotes_map = {c[0]: c[1] for c in clause_cits_quotes if c[1]}
        clean_text = _clean_citations(clause)
        if not clean_text:
            continue

        if cleanup:
            if _is_meta_claim(clean_text) or _is_refusal_sentence(clean_text):
                continue
            if not _has_predicate(clean_text):
                continue
            if len(clean_text.split()) < 3:
                continue

        if idx > 0 and subject:
            starts_with_verb = re.match(
                r"^(?:operates|provides|requires|holds|states|warrants|agrees|is|was|has|shall|will|must|may|can|includes|governs|specifies)\b",
                clean_text,
                re.IGNORECASE,
            )
            if starts_with_verb:
                clean_text = f"{subject} {clean_text}"

        clean_text = clean_text[0].upper() + clean_text[1:]
        if not clean_text.endswith((".", "!", "?")):
            clean_text += "."

        final_cits = clause_cits if clause_cits else list(sent_cits)
        final_quotes = clause_quotes_map if clause_quotes_map else dict(sent_quotes_map)
        primary_quote = next(iter(final_quotes.values()), None)
        results.append((clean_text, final_cits, primary_quote, final_quotes))

    return results


def extract_claims_rule_based(
    answer: str,
    query: Optional[str] = None,
    cleanup: bool = False,
) -> ClaimExtractionResult:
    """Decompose answer into atomic propositions using deterministic heuristics."""
    if not answer or not answer.strip():
        return ClaimExtractionResult(
            query=query,
            draft_answer=answer or "",
            claims=[],
            total_claims=0,
            cited_claims_count=0,
            uncited_claims_count=0,
            extraction_method="rule_based",
        )

    sentences = _split_into_sentences_bracket_aware(answer) if cleanup else _split_into_sentences(answer)
    extracted_claims: List[ExtractedClaim] = []
    claim_counter = 1

    for sentence in sentences:
        if _is_refusal_sentence(sentence):
            continue
        if cleanup and _is_meta_claim(sentence):
            continue

        decomposed = _decompose_sentence(sentence, cleanup=cleanup)
        for item in decomposed:
            clean_text, cits, quote_val, quotes_map = item
            claim_id = f"claim_{claim_counter:03d}"
            claim_counter += 1
            extracted_claims.append(
                ExtractedClaim(
                    claim_id=claim_id,
                    claim_text=clean_text,
                    cited_chunk_ids=cits,
                    source_sentence=sentence,
                    is_cited=bool(cits),
                    quote=quote_val,
                    quotes=quotes_map,
                )
            )

    total = len(extracted_claims)
    cited_count = sum(1 for c in extracted_claims if c.is_cited)

    return ClaimExtractionResult(
        query=query,
        draft_answer=answer,
        claims=extracted_claims,
        total_claims=total,
        cited_claims_count=cited_count,
        uncited_claims_count=total - cited_count,
        extraction_method="rule_based",
    )


def _parse_llm_json(raw_text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract and parse a JSON array from LLM response text."""
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    if "```" in cleaned:
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    bracket_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", cleaned)
    if bracket_match:
        cleaned = bracket_match.group(0).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return None


def extract_claims_llm(
    answer: str,
    query: Optional[str] = None,
    llm_callable: Optional[Callable[[str, Optional[str]], str]] = None,
    cleanup: bool = False,
) -> ClaimExtractionResult:
    """Decompose answer into atomic propositions using an LLM callable or client."""
    if not answer or not answer.strip():
        return ClaimExtractionResult(
            query=query,
            draft_answer=answer or "",
            claims=[],
            total_claims=0,
            cited_claims_count=0,
            uncited_claims_count=0,
            extraction_method="llm",
        )

    allowed_citations = set(extract_citations_from_text(answer))

    if llm_callable is None:
        logger.warning("No LLM callable supplied. Falling back to rule-based extractor.")
        return extract_claims_rule_based(answer, query=query, cleanup=cleanup)

    user_prompt = (
        f"USER QUESTION: {query or 'N/A'}\n\n"
        f"DRAFT ANSWER TO DECOMPOSE:\n{answer}"
    )

    try:
        raw_output = llm_callable(user_prompt, DEFAULT_CLAIM_EXTRACTION_SYSTEM_PROMPT)
    except Exception as e:
        logger.warning("LLM call failed (%s). Falling back to rule-based extractor.", e)
        return extract_claims_rule_based(answer, query=query, cleanup=cleanup)

    parsed_items = _parse_llm_json(raw_output)
    if parsed_items is None:
        logger.warning("LLM response could not be parsed as JSON. Falling back to rule-based extractor.")
        return extract_claims_rule_based(answer, query=query, cleanup=cleanup)

    extracted_claims: List[ExtractedClaim] = []

    for item in parsed_items:
        if not isinstance(item, dict):
            return extract_claims_rule_based(answer, query=query, cleanup=cleanup)

        raw_claim_text = item.get("claim") or item.get("claim_text")
        if not raw_claim_text or not isinstance(raw_claim_text, str) or not raw_claim_text.strip():
            return extract_claims_rule_based(answer, query=query, cleanup=cleanup)

        cited_chunks = item.get("cited_chunks") or item.get("cited_chunk_ids") or []
        if not isinstance(cited_chunks, list):
            return extract_claims_rule_based(answer, query=query, cleanup=cleanup)

        # CITATION INVARIANT: check that cited IDs literally occurred in input
        for cid in cited_chunks:
            if cid not in allowed_citations:
                logger.warning(
                    "LLM invented citation '%s' (allowed: %s). Falling back to rule-based extractor.",
                    cid,
                    allowed_citations,
                )
                return extract_claims_rule_based(answer, query=query, cleanup=cleanup)

        clean_text = _clean_citations(raw_claim_text)
        if not clean_text.endswith((".", "!", "?")):
            clean_text += "."

        if cleanup:
            if _is_meta_claim(clean_text) or _is_refusal_sentence(clean_text):
                continue
            if not _has_predicate(clean_text):
                continue
            if len(clean_text.split()) < 3:
                continue

        source_sentence = item.get("source_sentence") or answer
        quote_val = item.get("quote")
        quotes_map: Dict[str, str] = {}
        for cid, q in extract_citations_and_quotes_from_text(source_sentence or raw_claim_text):
            if q and cid in cited_chunks:
                quotes_map[cid] = q
        if not quote_val and quotes_map:
            quote_val = next(iter(quotes_map.values()), None)

        extracted_claims.append(
            ExtractedClaim(
                claim_id="temp",
                claim_text=clean_text,
                cited_chunk_ids=cited_chunks,
                source_sentence=source_sentence,
                is_cited=bool(cited_chunks),
                quote=quote_val,
                quotes=quotes_map,
            )
        )

    for i, claim in enumerate(extracted_claims, 1):
        claim.claim_id = f"claim_{i:03d}"

    total = len(extracted_claims)
    cited_count = sum(1 for c in extracted_claims if c.is_cited)

    return ClaimExtractionResult(
        query=query,
        draft_answer=answer,
        claims=extracted_claims,
        total_claims=total,
        cited_claims_count=cited_count,
        uncited_claims_count=total - cited_count,
        extraction_method="llm",
    )


def extract_claims(
    input_data: Union[str, Any],
    query: Optional[str] = None,
    llm_callable: Optional[Callable[[str, Optional[str]], str]] = None,
    method: str = "auto",
    cleanup: bool = False,
) -> ClaimExtractionResult:
    """Primary claim extraction entry point."""
    if hasattr(input_data, "answer"):
        answer = getattr(input_data, "answer")
        if query is None and hasattr(input_data, "query"):
            query = getattr(input_data, "query")
    else:
        answer = str(input_data)

    method_lower = method.lower().strip()
    if method_lower == "rule_based":
        return extract_claims_rule_based(answer, query=query, cleanup=cleanup)
    elif method_lower == "mock":
        res = extract_claims_rule_based(answer, query=query, cleanup=cleanup)
        res.extraction_method = "mock"
        return res
    elif method_lower == "llm":
        return extract_claims_llm(answer, query=query, llm_callable=llm_callable, cleanup=cleanup)
    elif method_lower == "auto":
        if llm_callable is not None:
            return extract_claims_llm(answer, query=query, llm_callable=llm_callable, cleanup=cleanup)
        return extract_claims_rule_based(answer, query=query, cleanup=cleanup)
    else:
        raise ValueError(f"Unknown extraction method: '{method}'. Expected 'auto', 'llm', 'rule_based', or 'mock'.")

