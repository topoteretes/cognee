"""
Test cognee-skills via the Python SDK — for developers building custom workflows.

This is the simplest way to use cognee-skills programmatically.
No MCP, no server — just import and call.

Prerequisites:
    pip install cognee
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SKILLS_DIR = Path(__file__).parent / "skills"

import cognee  # noqa: E402
from cognee import skills  # noqa: E402


async def main():
    # Clean slate
    await cognee.prune.prune_system()

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Ingest
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 1: Ingest skills ===")
    await skills.ingest(SKILLS_DIR)

    all_skills = await skills.list()
    for s in all_skills:
        print(f"  {s['skill_id']}: {s.get('instruction_summary', '')[:80]}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Run the skill 3 times (bad output each time)
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 2: Run the skill 3 times (bad instructions -> low quality) ===")

    tasks = [
        "Write a status update for the platform team. We shipped the new search API, "
        "fixed a caching bug, and started work on the admin dashboard.",
        "Write an incident report. The search API was down for 20 minutes "
        "because of a bad regex in the query parser. Alice found it, Bob deployed the fix.",
        "Write an announcement that we're moving to biweekly releases starting March 24.",
    ]

    for i, task in enumerate(tasks, 1):
        result = await skills.run(task, auto_amendify=False)
        print(f"\n  Run {i}:")
        print(f"    Quality score: {result.get('quality_score', 'N/A')}")
        print(f"    Output: {result['output'][:150]}...")

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Inspect
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 3: Inspect the skill ===")

    inspection = await skills.inspect(
        all_skills[0]["skill_id"],
        min_runs=1,
        score_threshold=0.8,
    )

    if inspection is None:
        print("  No failures detected")
    else:
        print(f"  Failure category: {inspection['failure_category']}")
        print(f"  Root cause:       {inspection['root_cause']}")
        print(f"  Severity:         {inspection['severity']}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: Preview the fix
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 4: Preview the fix ===")

    amendment = await skills.preview_amendify(
        all_skills[0]["skill_id"],
        min_runs=1,
        score_threshold=0.8,
    )

    if amendment is None:
        print("  No amendment proposed")
    else:
        print(f"  Amendment ID:  {amendment['amendment_id']}")
        print(f"  Explanation:   {amendment['change_explanation']}")
        print(f"  Confidence:    {amendment['amendment_confidence']}")

        print("\n  --- ORIGINAL INSTRUCTIONS ---")
        print(amendment["original_instructions"])
        print("\n  --- PROPOSED INSTRUCTIONS ---")
        print(amendment["amended_instructions"])

        # ──────────────────────────────────────────────────────────────
        # STEP 5: Apply the fix
        # ──────────────────────────────────────────────────────────────
        print("\n=== STEP 5: Apply the fix ===")

        apply_result = await skills.amendify(amendment["amendment_id"])
        print(f"  Applied: {apply_result.get('success', False)}")
        print(f"  Status:  {apply_result.get('status', '?')}")

        # ──────────────────────────────────────────────────────────────
        # STEP 6: Run again with the fixed skill
        # ──────────────────────────────────────────────────────────────
        print("\n=== STEP 6: Run the same task again (after fix) ===")

        result_after = await skills.run(tasks[0])
        print(f"  Quality score: {result_after.get('quality_score', 'N/A')}")
        print(f"  Output:\n{result_after['output']}")

    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
