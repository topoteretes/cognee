"""LLM-as-a-Judge Tiebreaker Engine.

Provides an optional semantic arbitrator for ambiguous NLI verdicts.
Requires verbatim quotes for entailment verdicts and tracks token/cost metrics.
"""

from __future__ import annotations

from enum import Enum
import json
import logging
import re
import time
from typing import Any, Optional

from pydantic import BaseModel

from examples.cookbooks.trustrag.core.quote_validator import validate_quote_in_text

logger = logging.getLogger(__name__)

INPUT_TOKEN_COST_PER_M = 0.075
OUTPUT_TOKEN_COST_PER_M = 0.30

LLM_JUDGE_SYSTEM_PROMPT = """You are an impartial, highly rigorous factual verification judge for financial and legal disclosures.
Your task is to judge whether a given atomic claim is fully supported, contradicted, or neutral/unsupported based SOLELY on the provided evidence chunk.

CRITICAL RULES:
1. SUPPORTED (entailment): The evidence explicitly and directly substantiates every factual assertion in the claim.
   You MUST provide a short, exact, verbatim quote from the evidence that proves the claim.
2. CONTRADICTED: The evidence directly contradicts or refutes the claim (e.g. wrong dates, numbers, parties, or polarities).
   You MUST provide a short, exact, verbatim quote from the evidence showing the contradiction.
3. NEUTRAL / UNSUPPORTED: The evidence does not contain sufficient facts to prove or disprove the claim.
   Quote should be null.

STRICT JSON OUTPUT FORMAT ONLY:
{
  "verdict": "entailment" | "contradiction" | "neutral",
  "verbatim_quote": "Exact verbatim quote from evidence chunk or null",
  "reasoning": "Concise factual rationale (1-2 sentences)"
}
"""


class LLMJudgeVerdict(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class LLMJudgeResult(BaseModel):
    """Container for LLM judge arbitration, quote verification, and cost transparency."""

    enabled: bool = False
    verdict: LLMJudgeVerdict = LLMJudgeVerdict.NEUTRAL
    verbatim_quote: Optional[str] = None
    quote_valid: bool = False
    quote_match_type: Optional[str] = None
    reasoning: str = ""
    model: str = "gemini-2.5-flash"
    latency_ms: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_usd: float = 0.0
    safety_override_applied: bool = False
    safety_override_reason: Optional[str] = None


def evaluate_llm_judge(
    claim_text: str,
    chunk_text: str,
    nli_verdict: str,
    llm_client: Optional[Any] = None,
    fact_check_passed: bool = True,
    quote_valid: bool = True,
    enabled: bool = False,
    model_name: str = "gemini-2.5-flash",
) -> LLMJudgeResult:
    """Invoke LLM judge tiebreaker with verbatim quote validation and safety overrides."""
    if not enabled or llm_client is None:
        return LLMJudgeResult(
            enabled=False,
            verdict=LLMJudgeVerdict(nli_verdict) if nli_verdict in LLMJudgeVerdict._value2member_map_ else LLMJudgeVerdict.NEUTRAL,
            reasoning="LLM judge disabled or no client provided.",
        )

    if not fact_check_passed:
        return LLMJudgeResult(
            enabled=True,
            verdict=LLMJudgeVerdict.CONTRADICTION,
            reasoning="Fact check failed (number/date mismatch); LLM judge bypassed by safety invariant.",
            safety_override_applied=True,
            safety_override_reason="Fact check failure takes absolute precedence.",
        )
    if not quote_valid:
        return LLMJudgeResult(
            enabled=True,
            verdict=LLMJudgeVerdict.NEUTRAL,
            reasoning="Quote validation failed; LLM judge bypassed by safety invariant.",
            safety_override_applied=True,
            safety_override_reason="Invalid quote grounding prevents judge entailment.",
        )

    user_prompt = (
        f"EVIDENCE CHUNK:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
        f"ATOMIC CLAIM TO VERIFY:\n\"\"\"\n{claim_text}\n\"\"\"\n\n"
        "Judge whether this claim is entailment, contradiction, or neutral. "
        "Return valid JSON only."
    )

    t0 = time.perf_counter()
    tokens_in = len(user_prompt.split()) + len(LLM_JUDGE_SYSTEM_PROMPT.split())
    tokens_out = 50

    try:
        if hasattr(llm_client, "generate"):
            response_text = llm_client.generate(
                prompt=user_prompt,
                system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
            )
        elif hasattr(llm_client, "complete"):
            res = llm_client.complete(
                prompt=user_prompt,
                system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
            )
            response_text = res.content if hasattr(res, "content") else str(res)
        elif callable(llm_client):
            response_text = llm_client(user_prompt, LLM_JUDGE_SYSTEM_PROMPT)
        else:
            raise AttributeError("llm_client must be callable or provide generate()/complete()")
    except Exception as exc:
        logger.error("LLM judge invocation failed: %s", exc)
        return LLMJudgeResult(
            enabled=True,
            verdict=LLMJudgeVerdict(nli_verdict) if nli_verdict in LLMJudgeVerdict._value2member_map_ else LLMJudgeVerdict.NEUTRAL,
            reasoning=f"LLM judge call error: {exc}",
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned_json)
        raw_verdict = str(data.get("verdict", "neutral")).lower().strip()
        v_quote = data.get("verbatim_quote")
        reasoning = str(data.get("reasoning", ""))
    except Exception:
        raw_verdict = "neutral"
        v_quote = None
        reasoning = response_text[:200]

    cost_usd = round(
        (tokens_in / 1_000_000.0) * INPUT_TOKEN_COST_PER_M
        + (tokens_out / 1_000_000.0) * OUTPUT_TOKEN_COST_PER_M,
        6,
    )

    quote_matches = False
    match_type = None
    if v_quote and len(v_quote.strip()) >= 10:
        matches, _, m_type = validate_quote_in_text(v_quote, chunk_text)
        quote_matches = matches
        match_type = m_type

    safety_override = False
    safety_reason = None
    final_v = raw_verdict

    if raw_verdict == "entailment":
        if not v_quote or not quote_matches:
            final_v = "neutral"
            safety_override = True
            safety_reason = "Judge returned entailment but failed to provide a valid verbatim quote from the chunk."

    if final_v not in LLMJudgeVerdict._value2member_map_:
        final_v = "neutral"

    return LLMJudgeResult(
        enabled=True,
        verdict=LLMJudgeVerdict(final_v),
        verbatim_quote=v_quote,
        quote_valid=quote_matches,
        quote_match_type=match_type,
        reasoning=reasoning,
        model=model_name,
        latency_ms=latency_ms,
        tokens_prompt=tokens_in,
        tokens_completion=tokens_out,
        cost_usd=cost_usd,
        safety_override_applied=safety_override,
        safety_override_reason=safety_reason,
    )

