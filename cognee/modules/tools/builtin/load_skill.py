"""Progressive-disclosure primitive: return a skill's procedure body by name.

The AgenticRetriever loads only Skill descriptions into the system prompt; the
full procedure is retrieved by the LLM calling this tool when it decides a
skill is relevant. Active skills for the current turn live in a ContextVar so
the handler does not need another graph round-trip.
"""

from typing import Any, Dict

from cognee.modules.engine.models import Tool
from cognee.modules.tools.context import active_skills_var, opened_skills_var
from cognee.modules.tools.errors import ToolInvocationError
from cognee.modules.tools.registry import register_builtin_tool


MAX_SKILL_BODY_CHARS = 12_000

TOOL = Tool(
    name="load_skill",
    description=(
        "Load the full procedure body of a named skill. Use this when the skill "
        "description suggests the skill applies to the current task."
    ),
    input_schema={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Name of the skill to load."}},
        "required": ["name"],
    },
    handler_ref="cognee.modules.tools.builtin.load_skill.handler",
    permission_required="read",
    readonly_hint=True,
)


async def handler(args: Dict[str, Any], **_) -> str:
    name = args.get("name")
    if not name:
        raise ToolInvocationError("load_skill requires a 'name' argument")

    skills = active_skills_var.get()
    skill = skills.get(name)
    if skill is None:
        available = ", ".join(sorted(skills)) or "(none in scope)"
        raise ToolInvocationError(
            f"Skill {name!r} is not in the active skill set. Available: {available}"
        )

    opened = opened_skills_var.get()
    if opened is not None:
        opened.add(skill.name)

    body = skill.procedure or "(this skill has no procedure body)"
    if len(body) > MAX_SKILL_BODY_CHARS:
        body = body[:MAX_SKILL_BODY_CHARS] + "\n... [truncated]"
    return f"# Skill: {skill.name}\n{body}"


register_builtin_tool(TOOL)
