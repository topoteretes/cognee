"""Binary LLM-judge accuracy for LoCoMo (the metric mem0 / Zep style reports quote).

The judge sees the question, the gold answer and the model answer and returns CORRECT or
WRONG. It runs on its own model (``LOCOMO_JUDGE_MODEL``, default ``openai/gpt-5.1``) through
litellm directly, so the answering model configured for cognee (``LLM_MODEL``) is never the
one grading itself. The API key falls back to cognee's ``LLM_API_KEY``.

Adversarial questions are graded on abstention: the answer is CORRECT when it says the
information is not in the conversation, and WRONG when it commits to a specific answer —
in particular the dataset's ``adversarial_answer`` distractor.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger()

DEFAULT_JUDGE_MODEL = "openai/gpt-5.1"
JUDGE_MODEL_ENV = "LOCOMO_JUDGE_MODEL"
JUDGE_RETRIES = 4
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "locomo_judge_prompt.txt"
_LABEL_RE = re.compile(r"\b(CORRECT|WRONG)\b", re.IGNORECASE)


def judge_model_name() -> str:
    return os.getenv(JUDGE_MODEL_ENV) or DEFAULT_JUDGE_MODEL


def _judge_api_key() -> str | None:
    explicit = os.getenv("LOCOMO_JUDGE_API_KEY")
    if explicit:
        return explicit
    try:
        from cognee.infrastructure.llm.config import get_llm_config

        return get_llm_config().llm_api_key
    except Exception:  # pragma: no cover - config import failure is environmental
        return os.getenv("LLM_API_KEY")


def load_judge_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def render_judge_prompt(
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    question_type: str,
    adversarial_answer: str | None = None,
) -> str:
    template = load_judge_prompt()
    if question_type == "adversarial":
        gold_block = (
            "This question is UNANSWERABLE from the conversation. The correct behaviour is to "
            "say the information is not available. A response that commits to a concrete answer "
            "is WRONG"
        )
        if adversarial_answer:
            gold_block += f' — especially the plausible-sounding distractor "{adversarial_answer}"'
        gold_block += "."
    else:
        gold_block = f"Gold answer: {gold_answer}"
    return (
        template.replace("{question}", question)
        .replace("{gold_block}", gold_block)
        .replace("{response}", model_answer if model_answer else "(empty response)")
        .replace("{question_type}", question_type)
    )


def parse_judge_output(raw: str) -> dict[str, Any]:
    """Accept ``{"label": ..., "reason": ...}`` JSON or a bare CORRECT/WRONG verdict."""
    text = (raw or "").strip()
    if not text:
        return {"label": None, "reason": "empty judge output"}

    candidate = text
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            candidate = brace.group(0)

    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            label = str(payload.get("label", "")).strip().upper()
            if label in {"CORRECT", "WRONG"}:
                return {"label": label, "reason": str(payload.get("reason", "")).strip()}
    except json.JSONDecodeError:
        pass

    match = _LABEL_RE.search(text)
    if match:
        return {"label": match.group(1).upper(), "reason": text[:500]}
    return {"label": None, "reason": f"unparseable judge output: {text[:200]}"}


async def call_judge(prompt: str, *, model: str | None = None) -> str:
    import litellm

    model = model or judge_model_name()
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(JUDGE_RETRIES):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict but fair grader. Reply with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                api_key=_judge_api_key(),
                max_completion_tokens=400,
            )
            return response.choices[0].message.content or ""
        except Exception as error:  # network / rate-limit / transient provider errors
            last_error = error
            logger.warning("LoCoMo judge call failed (attempt %s): %s", attempt + 1, error)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 16.0)
    raise RuntimeError(f"LoCoMo judge failed after {JUDGE_RETRIES} attempts: {last_error}")


class LocomoLLMJudgeMetric:
    """Async metric: ``score`` is 1.0 for CORRECT, 0.0 for WRONG, ``None`` when unparseable."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or judge_model_name()
        self.score: float | None = None
        self.reason: str | None = None

    async def a_measure(self, test_case: Any) -> float | None:
        metadata = getattr(test_case, "additional_metadata", None) or {}
        question_type = str(metadata.get("question_type", "unknown"))
        prompt = render_judge_prompt(
            question=test_case.input,
            gold_answer=test_case.expected_output or "",
            model_answer=test_case.actual_output or "",
            question_type=question_type,
            adversarial_answer=metadata.get("adversarial_answer"),
        )
        raw = await call_judge(prompt, model=self.model)
        verdict = parse_judge_output(raw)
        label = verdict["label"]
        self.score = None if label is None else (1.0 if label == "CORRECT" else 0.0)
        self.reason = f"[{self.model}] {label}: {verdict['reason']}"
        return self.score

    def measure(self, test_case: Any) -> float | None:
        return asyncio.run(self.a_measure(test_case))
