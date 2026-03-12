"""
Test cognee-skills via MCP — the way Claude Code and other MCP IDEs use it.

This script connects to the cognee-mcp server over stdio (same as Claude Code),
calls the MCP tools directly, and shows the full self-improvement loop.

Prerequisites:
    pip install cognee-mcp mcp
"""

import asyncio
import json
import os

from dotenv import load_dotenv
load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_tool(session: ClientSession, name: str, args: dict):
    """Call an MCP tool and return parsed JSON result (or raw text if not JSON)."""
    result = await session.call_tool(name, args)
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def main():
    # Connect to cognee-mcp over stdio — exactly how Claude Code does it
    server_params = StdioServerParameters(
        command="cognee-mcp",
        env={**os.environ},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Show available tools
            tools = await session.list_tools()
            print(f"Connected to cognee-mcp ({len(tools.tools)} tools available)")

            # ──────────────────────────────────────────────────────────────
            # STEP 1: Clean slate + ingest
            # ──────────────────────────────────────────────────────────────
            print("\n=== STEP 1: Ingest skills ===")

            prune_result = await call_tool(session, "prune", {})
            print(f"  Pruned: {prune_result}" if isinstance(prune_result, str) else "  Pruned")

            ingest_result = await call_tool(session, "ingest_skills", {
                "skills_folder": "./skills",
            })
            if isinstance(ingest_result, str):
                print(f"  {ingest_result}")
            else:
                print(f"  Ingested: {ingest_result.get('skill_count', '?')} skills")

            skill_list = await call_tool(session, "list_skills", {})
            if isinstance(skill_list, list):
                for s in skill_list:
                    print(f"    - {s['skill_id']}")
            else:
                print(f"  Skills: {skill_list}")

            # ──────────────────────────────────────────────────────────────
            # STEP 2: Run the skill 3 times (bad output each time)
            # ──────────────────────────────────────────────────────────────
            print("\n=== STEP 2: Run the skill 3 times (bad instructions -> low quality) ===")

            tasks = [
                "Write a status update for the platform team. We shipped the new "
                "search API, fixed a caching bug, and started work on the admin dashboard.",

                "Write an incident report. The search API was down for 20 minutes "
                "because of a bad regex in the query parser. Alice found it, Bob deployed the fix.",

                "Write an announcement that we're moving to biweekly releases starting March 24.",
            ]

            for i, task in enumerate(tasks, 1):
                result = await call_tool(session, "run_skill", {
                    "task_text": task,
                    "auto_amendify": False,
                })
                print(f"\n  Run {i}:")
                if isinstance(result, dict):
                    print(f"    Quality: {result.get('quality_score', 'N/A')}")
                    print(f"    Output:  {str(result.get('output', ''))[:120]}...")
                else:
                    print(f"    Result: {str(result)[:120]}...")

            # ──────────────────────────────────────────────────────────────
            # STEP 3: Inspect
            # ──────────────────────────────────────────────────────────────
            print("\n=== STEP 3: Inspect the skill ===")

            skill_id = skill_list[0]["skill_id"] if isinstance(skill_list, list) else "team-comms"
            inspection = await call_tool(session, "inspect_skill", {
                "skill_id": skill_id,
                "min_runs": 1,
                "score_threshold": 0.8,
            })

            if isinstance(inspection, dict):
                print(f"  Category:   {inspection.get('failure_category')}")
                print(f"  Root cause: {inspection.get('root_cause', '')[:200]}")
                print(f"  Severity:   {inspection.get('severity')}")
            else:
                print(f"  {inspection}")

            # ──────────────────────────────────────────────────────────────
            # STEP 4: Preview the fix
            # ──────────────────────────────────────────────────────────────
            print("\n=== STEP 4: Preview the fix ===")

            amendment = await call_tool(session, "preview_amendify_skill", {
                "skill_id": skill_id,
                "min_runs": 1,
                "score_threshold": 0.8,
            })

            amendment_id = None
            if isinstance(amendment, dict):
                amendment_id = amendment.get("amendment_id")
                print(f"  Amendment:   {amendment_id}")
                print(f"  Explanation: {amendment.get('change_explanation', '')[:200]}")
                print(f"  Confidence:  {amendment.get('amendment_confidence')}")

                print(f"\n  --- ORIGINAL INSTRUCTIONS ---")
                print(amendment.get('original_instructions', 'N/A'))
                print(f"\n  --- PROPOSED INSTRUCTIONS ---")
                print(amendment.get('amended_instructions', 'N/A'))
            else:
                print(f"  {amendment}")

            if amendment_id:
                # ──────────────────────────────────────────────────────────
                # STEP 5: Apply the fix
                # ──────────────────────────────────────────────────────────
                print("\n=== STEP 5: Apply the fix ===")

                apply_result = await call_tool(session, "amendify_skill", {
                    "amendment_id": amendment_id,
                })
                if isinstance(apply_result, dict):
                    print(f"  Applied: {apply_result.get('success')}")
                else:
                    print(f"  {apply_result}")

                # ──────────────────────────────────────────────────────────
                # STEP 6: Run again with the fixed skill
                # ──────────────────────────────────────────────────────────
                print("\n=== STEP 6: Run the same task again (after fix) ===")

                result_after = await call_tool(session, "run_skill", {
                    "task_text": tasks[0],
                })
                if isinstance(result_after, dict):
                    print(f"  Quality: {result_after.get('quality_score', 'N/A')}")
                    print(f"  Output:\n{result_after.get('output', '')}")
                else:
                    print(f"  {result_after}")

            print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
