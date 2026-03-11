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
