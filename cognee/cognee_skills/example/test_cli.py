"""
Test cognee-skills via the CLI — for terminal users.

Runs the same self-improvement loop as the other tests,
but through `cognee-cli skills` commands.

Prerequisites:
    pip install cognee
"""

import json
import subprocess
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"


def run(cmd: str) -> str:
    """Run a CLI command and return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 and "Error" in (result.stderr or ""):
        print(f"  [stderr] {result.stderr.strip()[:200]}")
    return result.stdout.strip()


def run_json(cmd: str) -> dict | list:
    """Run a CLI command with --output-format json and parse the result."""
    out = run(cmd)
    return json.loads(out)


def main():
    # ──────────────────────────────────────────────────────────────────
    # STEP 0: Clean slate
    # ──────────────────────────────────────────────────────────────────
    print("=== STEP 0: Prune ===")
    run("cognee-cli delete --all")
    print("  Pruned")

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Ingest
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 1: Ingest skills ===")
    print(f"  {run(f'cognee-cli skills ingest {SKILLS_DIR}')}")

    skills_list = run_json("cognee-cli skills list -f json")
    for s in skills_list:
        print(f"  Ingested: {s['skill_id']}")

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
        result = run_json(f'cognee-cli skills run "{task}" --no-amendify -f json')
        print(f"\n  Run {i}:")
        print(f"    Quality: {result.get('quality_score', 'N/A')}")
        print(f"    Output:  {str(result.get('output', ''))[:120]}...")

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Inspect
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 3: Inspect the skill ===")

    skill_id = skills_list[0]["skill_id"]
    inspection = run_json(f"cognee-cli skills inspect {skill_id} --score-threshold 0.8 -f json")
    print(f"  Category:   {inspection.get('failure_category')}")
    print(f"  Root cause: {inspection.get('root_cause', '')[:200]}")
    print(f"  Severity:   {inspection.get('severity')}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: Preview the fix
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 4: Preview the fix ===")

    amendment = run_json(f"cognee-cli skills preview {skill_id} --score-threshold 0.8 -f json")
    amendment_id = amendment.get("amendment_id")
    print(f"  Amendment:   {amendment_id}")
    print(f"  Explanation: {amendment.get('change_explanation', '')[:200]}")
    print(f"  Confidence:  {amendment.get('amendment_confidence')}")

    print("\n  --- ORIGINAL INSTRUCTIONS ---")
    print(amendment.get("original_instructions", "N/A"))
    print("\n  --- PROPOSED INSTRUCTIONS ---")
    print(amendment.get("amended_instructions", "N/A"))

    # ──────────────────────────────────────────────────────────────────
    # STEP 5: Apply the fix
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 5: Apply the fix ===")
    print(f"  {run(f'cognee-cli skills amendify {amendment_id}')}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 6: Run again with the fixed skill
    # ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 6: Run the same task again (after fix) ===")

    result_after = run_json(f'cognee-cli skills run "{tasks[0]}" -f json')
    print(f"  Quality: {result_after.get('quality_score', 'N/A')}")
    print(f"  Output:\n{result_after.get('output', '')}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
