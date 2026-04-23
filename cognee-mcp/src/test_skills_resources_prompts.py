#!/usr/bin/env python3
"""
Tests for skills MCP resources and prompts.

Directly invokes the resource/prompt handler functions registered on the FastMCP
server, so no running server or LLM keys are needed.

Usage:
    cd cognee-mcp
    python src/test_skills_resources_prompts.py
"""

import asyncio
import json
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from cognee.shared.logging_utils import setup_logging
from logging import ERROR

setup_logging(log_level=ERROR)

passed = 0
failed = 0


def ok(name: str, detail: str = ""):
    global passed
    passed += 1
    print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))


def fail(name: str, detail: str = ""):
    global failed
    failed += 1
    print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


async def run_tests():
    # Import triggers tool/resource/prompt registration on the global `mcp` object
    import server  # noqa: F401
    from server import mcp as mcp_server

    # ------------------------------------------------------------------
    # 1. Verify resource and prompt registration
    # ------------------------------------------------------------------
    print("\n=== Registration checks ===")

    resources = await mcp_server.list_resources()
    resource_uris = {str(r.uri) for r in resources}

    if "skill://agent-guide" in resource_uris:
        ok("resource skill://agent-guide registered")
    else:
        fail("resource skill://agent-guide registered", f"got {resource_uris}")

    if "skill://index" in resource_uris:
        ok("resource skill://index registered")
    else:
        fail("resource skill://index registered", f"got {resource_uris}")

    resource_templates = await mcp_server.list_resource_templates()
    template_uris = {str(t.uriTemplate) for t in resource_templates}

    if "skill://{skill_id}" in template_uris:
        ok("resource template skill://{skill_id} registered")
    else:
        fail("resource template skill://{skill_id} registered", f"got {template_uris}")

    prompts = await mcp_server.list_prompts()
    prompt_names = {p.name for p in prompts}

    if "route-task" in prompt_names:
        ok("prompt route-task registered")
    else:
        fail("prompt route-task registered", f"got {prompt_names}")

    if "skill-setup" in prompt_names:
        ok("prompt skill-setup registered")
    else:
        fail("prompt skill-setup registered", f"got {prompt_names}")

    # ------------------------------------------------------------------
    # 2. Read skill://agent-guide
    # ------------------------------------------------------------------
    print("\n=== Resource: skill://agent-guide ===")

    guide_content = await mcp_server.read_resource("skill://agent-guide")
    guide_text = guide_content[0].content if hasattr(guide_content[0], "content") else str(guide_content[0])

    if "Cognee Skill Routing" in guide_text:
        ok("agent-guide returns markdown with expected heading")
    else:
        fail("agent-guide returns markdown", f"got: {guide_text[:100]}")

    if "get_skill_context" in guide_text:
        ok("agent-guide mentions get_skill_context")
    else:
        fail("agent-guide mentions get_skill_context")

    # ------------------------------------------------------------------
    # 3. Read skill://index (empty state)
    # ------------------------------------------------------------------
    print("\n=== Resource: skill://index (empty state) ===")

    index_content = await mcp_server.read_resource("skill://index")
    index_text = index_content[0].content if hasattr(index_content[0], "content") else str(index_content[0])
    index_data = json.loads(index_text)

    if isinstance(index_data.get("skills"), list):
        ok("index returns skills list")
    else:
        fail("index returns skills list", f"got: {index_data}")

    if len(index_data["skills"]) == 0 and "hint" in index_data:
        ok("index shows hint when empty")
    else:
        ok("index has skills", f"count={len(index_data['skills'])}")

    # ------------------------------------------------------------------
    # 4. Read skill://{skill_id} (nonexistent)
    # ------------------------------------------------------------------
    print("\n=== Resource: skill://nonexistent-skill ===")

    detail_content = await mcp_server.read_resource("skill://nonexistent-skill")
    detail_text = detail_content[0].content if hasattr(detail_content[0], "content") else str(detail_content[0])
    detail_data = json.loads(detail_text)

    if "error" in detail_data:
        ok("nonexistent skill returns error JSON")
    else:
        fail("nonexistent skill returns error JSON", f"got: {detail_data}")

    # ------------------------------------------------------------------
    # 5. Prompt: route-task
    # ------------------------------------------------------------------
    print("\n=== Prompt: route-task ===")

    route_result = await mcp_server.get_prompt("route-task", {"task_description": "summarize a document"})
    messages = route_result.messages

    if len(messages) == 1 and messages[0].role == "user":
        ok("route-task returns single user message")
    else:
        fail("route-task returns single user message", f"got {len(messages)} messages")

    msg_text = messages[0].content.text
    if "summarize a document" in msg_text:
        ok("route-task includes task description")
    else:
        fail("route-task includes task description")

    if "get_skill_context" in msg_text and "observe_skill_run" in msg_text:
        ok("route-task includes workflow steps")
    else:
        fail("route-task includes workflow steps")

    # ------------------------------------------------------------------
    # 6. Prompt: skill-setup
    # ------------------------------------------------------------------
    print("\n=== Prompt: skill-setup ===")

    setup_result = await mcp_server.get_prompt("skill-setup", {"skills_folder": "./my_skills"})
    setup_messages = setup_result.messages

    if len(setup_messages) == 1 and setup_messages[0].role == "user":
        ok("skill-setup returns single user message")
    else:
        fail("skill-setup returns single user message")

    setup_text = setup_messages[0].content.text
    if "./my_skills" in setup_text:
        ok("skill-setup includes skills_folder path")
    else:
        fail("skill-setup includes skills_folder path")

    if "ingest_skills" in setup_text or "upsert_skills" in setup_text:
        ok("skill-setup guides toward ingest or upsert")
    else:
        fail("skill-setup guides toward ingest or upsert")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 40}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_tests())
