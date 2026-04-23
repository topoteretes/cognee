"""
Test the meta-skill: cognee-skills' own SKILL.md ingested and executed.

The SKILL.md at the root of cognee_skills describes how to use cognee-skills
itself (run_skill, inspect_skill, amendify_skill, etc.).

This test ingests that SKILL.md as a skill and runs tasks against it to verify:
1. It routes correctly to the meta-skill
2. It produces useful, accurate guidance for skill-related tasks
3. The self-improvement loop works on the meta-skill itself
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import cognee
from cognee import skills


# Dedicated folder containing only the meta-skill SKILL.md
META_SKILL_DIR = Path(__file__).parent.parent / "meta-skill"


async def main():
    await cognee.prune.prune_system()

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Ingest the meta-skill
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 1: Ingest the meta-skill (cognee-skills' own SKILL.md) ===")

    await skills.ingest(str(META_SKILL_DIR))

    all_skills = await skills.list()
    for s in all_skills:
        print(f"  Ingested: {s['skill_id']} — {s.get('instruction_summary', '')[:80]}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Run tasks that the meta-skill should handle
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 2: Run tasks against the meta-skill ===")

    tasks = [
        "A skill keeps producing wrong output. How do I fix it?",
        "I want to see the proposed fix before applying it.",
        "The fix I applied made things worse. How do I revert it?",
    ]

    for i, task in enumerate(tasks, 1):
        result = await skills.run(task, auto_amendify=False)

        print(f"\n  Task {i}: {task}")
        print(f"  Skill routed to: {result.get('skill_id', '?')}")
        print(f"  Quality score:   {result.get('quality_score', 'N/A')}")
        print(f"  Output:\n{result.get('output', '')}")
        print()

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
