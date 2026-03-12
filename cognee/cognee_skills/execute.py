"""Execute a skill: load instructions, call LLM, return result with timing."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import litellm

from cognee.infrastructure.llm import get_llm_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are executing the skill "{skill_name}".

{instructions}

Follow the instructions above to complete the user's task. \
Be thorough but concise. If the instructions reference tools or external actions \
you cannot perform, describe what should be done instead."""

EVALUATE_PROMPT = """\
You are a quality evaluator. Score how useful the output is for the user's task.

The skill's instructions may be flawed — that's exactly what we're trying to detect. \
Do NOT score based on whether the output follows the instructions. \
Score based on whether a human reading this output would find it helpful for their task.

Task: {task_text}
Output to evaluate:
{output}

Respond with ONLY a JSON object: {{"score": <float 0.0 to 1.0>, "reason": "<one sentence>"}}

Scoring guide:
- 1.0: Output is exactly what someone would want for this task
- 0.7-0.9: Mostly useful, minor gaps
- 0.4-0.6: Partially useful, significant gaps
- 0.1-0.3: Mostly unhelpful or off-topic
- 0.0: Completely useless or empty"""


async def execute_skill(
    skill: Dict[str, Any],
    task_text: str,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a loaded skill against a task using the configured LLM.

    Args:
        skill: Skill dict as returned by Skills.load() (must have 'instructions' and 'name').
        task_text: The user's task description.
        context: Optional additional context to include in the user message.

    Returns:
        Dict with keys: output, model, latency_ms, success, error.
    """
    llm_config = get_llm_config()
    model = llm_config.llm_model
    api_key = llm_config.llm_api_key

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        skill_name=skill.get("name", "unknown"),
        instructions=skill.get("instructions", ""),
    )

    user_message = task_text
    if context:
        user_message = f"{task_text}\n\nAdditional context:\n{context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    start_ms = int(time.time() * 1000)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            api_key=api_key,
        )
        output = response.choices[0].message.content or ""
        latency_ms = int(time.time() * 1000) - start_ms

        logger.info(
            "Executed skill '%s' in %dms",
            skill.get("skill_id", ""),
            latency_ms,
        )

        return {
            "output": output,
            "skill_id": skill.get("skill_id", ""),
            "model": model,
            "latency_ms": latency_ms,
            "success": True,
            "error": None,
        }

    except Exception as exc:
        latency_ms = int(time.time() * 1000) - start_ms
        logger.warning(
            "Skill execution failed for '%s': %s",
            skill.get("skill_id", ""),
            exc,
        )
        return {
            "output": "",
            "skill_id": skill.get("skill_id", ""),
            "model": model,
            "latency_ms": latency_ms,
            "success": False,
            "error": str(exc),
        }


async def evaluate_output(
    skill: Dict[str, Any],
    task_text: str,
    output: str,
) -> Dict[str, Any]:
    """Score output quality with a second LLM call.

    Returns:
        Dict with keys: score (float 0.0-1.0), reason (str).
    """
    import json as _json

    llm_config = get_llm_config()

    prompt = EVALUATE_PROMPT.format(
        task_text=task_text,
        output=output[:2000],
    )

    try:
        response = await litellm.acompletion(
            model=llm_config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=llm_config.llm_api_key,
        )
        raw = response.choices[0].message.content or ""
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = _json.loads(cleaned)
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        reason = str(parsed.get("reason", ""))
        logger.info(
            "Evaluated skill '%s' output: score=%.2f reason=%s",
            skill.get("skill_id", ""),
            score,
            reason,
        )
        return {"score": score, "reason": reason}
    except Exception as exc:
        logger.warning("Output evaluation failed for '%s': %s", skill.get("skill_id", ""), exc)
        return {"score": 1.0, "reason": f"Evaluation failed: {exc}"}
