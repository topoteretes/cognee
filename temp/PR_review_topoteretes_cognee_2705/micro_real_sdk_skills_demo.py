"""Minimal real SDK demo for PR #2705 agentic skills.

Run from the repo root:

    uv run python temp/PR_review_topoteretes_cognee_2705/micro_real_sdk_skills_demo.py

This version uses the real Cognee SDK path. No monkeypatching, no fake storage,
no fake search backend.

Requirements:
- a working local Cognee DB/vector/graph setup
- LLM settings configured, because AGENTIC_COMPLETION calls the LLM

Current branch mapping:
- remember_skills(path) is folded into remember(path, content_type="skills")
- search(..., skills=[...]) exists through AGENTIC_COMPLETION
- improve_skills(...) is currently split:
  - proposal creation can be triggered through remember(SkillRunEntry, skill_improvement=...)
  - applying a reviewed proposal uses the internal improve_skill(..., apply=True) helper
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import cognee
from cognee import SearchType
from cognee.memory import SkillRunEntry
from cognee.modules.data.methods.get_authorized_dataset_by_name import (
    get_authorized_dataset_by_name,
)
from cognee.modules.engine.operations.setup import setup
from cognee.modules.memify.skill_improvement import improve_skill
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import ERROR, setup_logging


DATASET_NAME = "toy-agentic-skills"
DEMO_ROOT = Path("temp/PR_review_topoteretes_cognee_2705/_real_demo_skills")


def write_skill(root: Path, slug: str, description: str, body: str) -> None:
    skill_dir = root / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
description: {description}
allowed-tools: load_skill
---
# {slug}

{body}
""",
        encoding="utf-8",
    )


async def main() -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    DEMO_ROOT.mkdir(parents=True)

    write_skill(
        DEMO_ROOT,
        "echo-back",
        "Repeat short user text exactly.",
        "When asked to repeat text, return the same text exactly, including punctuation.",
    )
    write_skill(
        DEMO_ROOT,
        "reverse-text",
        "Reverse short user text.",
        "When asked to reverse text, return the characters in reverse order.",
    )

    remembered = await cognee.remember(
        str(DEMO_ROOT),
        dataset_name=DATASET_NAME,
        content_type="skills",
    )
    print(f"1. remember -> stored {remembered.items_processed} skills")

    answer = await cognee.search(
        "Use the echo-back skill. Repeat exactly: tomato!",
        query_type=SearchType.AGENTIC_COMPLETION,
        datasets=DATASET_NAME,
        skills=["echo-back"],
        max_iter=3,
        session_id="toy-agentic-skills-session",
    )
    print(f"2. search -> {answer}")

    user = await get_default_user()
    dataset = await get_authorized_dataset_by_name(DATASET_NAME, user, "write")

    proposal_result = await cognee.remember(
        SkillRunEntry(
            selected_skill_id="echo-back",
            task_text="Repeat exactly: tomato!",
            result_summary="Returned 'tomato' without punctuation.",
            success_score=0.2,
            feedback=-1.0,
        ),
        dataset_name=DATASET_NAME,
        session_id="toy-agentic-skills-session",
        skill_improvement={"skill_name": "echo-back", "apply": False},
    )
    proposal_id = next(
        item["proposal_id"]
        for item in proposal_result.items
        if item.get("kind") == "skill_improvement_proposal"
    )
    print(f"3. improve proposal -> proposal_id={proposal_id}")

    applied = await improve_skill(
        "echo-back",
        dataset=dataset,
        user=user,
        proposal_id=proposal_id,
        apply=True,
    )
    print(f"4. improve apply -> status={applied.status}")

    shutil.rmtree(DEMO_ROOT)


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
