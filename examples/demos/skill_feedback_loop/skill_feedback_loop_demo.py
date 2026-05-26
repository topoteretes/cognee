"""Small local demo for ingesting skills, recording a weak run, and proposing an improvement.

Run from the repo root:

    uv run python examples/demos/skill_feedback_loop/skill_feedback_loop_demo.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any
from uuid import UUID

os.environ["LOG_LEVEL"] = "ERROR"
os.environ["COGNEE_LOG_FILE"] = "false"
os.environ["COGNEE_CLI_MODE"] = "true"
warnings.filterwarnings("ignore", message="This declarative base already contains a class.*")

import cognee
from cognee import SearchType
from cognee.context_global_variables import set_database_global_context_variables
from cognee.memory import SkillRunEntry
from cognee.modules.engine.operations.setup import setup
from cognee.modules.memify.skill_improvement import improve_skill
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.tools.resolve_skills import find_skill_by_name


DATASET_NAME = "toy-skill-feedback-loop"
SESSION_ID = "toy-skill-feedback-loop-session"
DEMO_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = DEMO_ROOT / "skills"
DATA_ROOT = DEMO_ROOT / "data"
SKILL_NAMES = [
    "diff-risk-explainer",
    "pr-comment-evaluator",
    "skill-feedback-writer",
]
TASK_TEMPLATE = """Use the skills in this exact order:
1. Load diff-risk-explainer and explain the concrete bug risk in the diff.
2. Load pr-comment-evaluator and evaluate the reviewer comment.
3. Load skill-feedback-writer and decide which skill needs a better instruction.

The skills are plain instructions. After you load each skill, do the work yourself.
The pr-comment-evaluator skill is intentionally flawed because it judges tone only. If its output
does not compare the reviewer comment against the concrete bug risk, target pr-comment-evaluator
and give a score of 0.30 or lower.

Return only JSON with keys:
diff_risk_summary, comment_evaluation, skill_to_improve, score, feedback, missing_instruction.

Diff:
{diff_text}

Reviewer comment:
{comment_text}
"""


def _unwrap_answer(answer: Any) -> Any:
    if isinstance(answer, list) and answer:
        return _unwrap_answer(answer[0])
    if isinstance(answer, dict) and "search_result" in answer:
        return _unwrap_answer(answer["search_result"])
    return answer


def parse_json_answer(answer: Any) -> dict[str, Any]:
    text = _unwrap_answer(answer)
    if not isinstance(text, str):
        raise ValueError(f"Expected string answer, got {type(text).__name__}")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"Agent answer did not contain JSON: {text[:500]}") from None
        return json.loads(match.group(0))


def score_from_feedback(feedback: dict[str, Any]) -> float:
    score = float(feedback["score"])
    return max(0.0, min(1.0, score))


def one_line(body: str) -> str:
    return " ".join(body.split())


def feedback_summary(feedback: dict[str, Any]) -> str:
    return (
        f"Feedback: {feedback.get('feedback', '')}\n"
        f"Missing instruction: {feedback.get('missing_instruction', '')}\n"
        f"Diff risk summary: {feedback.get('diff_risk_summary', '')}\n"
        f"Comment evaluation: {feedback.get('comment_evaluation', '')}"
    )


async def skill_body(skill_name: str, dataset, user) -> str:
    owner_id = getattr(dataset, "owner_id", None) or getattr(user, "id", None)
    if owner_id is None:
        raise ValueError("skill_body requires a dataset owner or user id.")
    async with set_database_global_context_variables(dataset.id, owner_id):
        skill = await find_skill_by_name(skill_name, dataset_id=dataset.id)
    if skill is None:
        raise ValueError(f"Skill {skill_name!r} was not found.")
    return skill.procedure.strip()


async def main() -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    remembered = await cognee.remember(
        str(SKILLS_ROOT),
        dataset_name=DATASET_NAME,
        content_type="skills",
    )
    print(f"1. remember -> stored {remembered.items_processed} skills")
    user, datasets = await resolve_authorized_user_datasets(UUID(remembered.dataset_id))
    dataset = datasets[0]

    task = TASK_TEMPLATE.format(
        diff_text=(DATA_ROOT / "tiny_diff.patch").read_text(encoding="utf-8"),
        comment_text=(DATA_ROOT / "bad_pr_comment.txt").read_text(encoding="utf-8"),
    )
    answer = await cognee.search(
        task,
        query_type=SearchType.AGENTIC_COMPLETION,
        datasets=DATASET_NAME,
        skills=SKILL_NAMES,
        max_iter=6,
        session_id=SESSION_ID,
    )
    feedback = parse_json_answer(answer)
    score = score_from_feedback(feedback)
    skill_to_improve = str(feedback["skill_to_improve"])
    print(f"2. evaluation -> {skill_to_improve} scored {score:.2f}")

    proposal_result = await cognee.remember(
        SkillRunEntry(
            selected_skill_id=skill_to_improve,
            task_text=task,
            result_summary=feedback_summary(feedback),
            success_score=score,
            feedback=-1.0 if score < 0.7 else 1.0,
        ),
        dataset_name=DATASET_NAME,
        session_id=SESSION_ID,
        skill_improvement={
            "skill_name": skill_to_improve,
            "apply": False,
            "score_threshold": 0.9,
        },
    )
    proposal_id = next(
        item["proposal_id"]
        for item in proposal_result.items
        if item.get("kind") == "skill_improvement_proposal"
    )
    before = await skill_body(skill_to_improve, dataset, user)
    await improve_skill(
        skill_to_improve,
        dataset=dataset,
        user=user,
        proposal_id=proposal_id,
        apply=True,
    )
    after = await skill_body(skill_to_improve, dataset, user)
    print(f"3. improve proposal -> applied proposal_id={proposal_id}")
    print(f"4. skill before -> {one_line(before)}")
    print(f"5. skill after -> {one_line(after)}")


if __name__ == "__main__":
    asyncio.run(main())
