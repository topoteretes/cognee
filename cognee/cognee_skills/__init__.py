"""cognee.cognee_skills — skill runtime (routing, execution, self-improvement).

Skill ingestion lives on ``cognee.remember(path, enrich=...)`` — this
package is the runtime side: finding the best skill for a task,
executing it, scoring the output, and self-repairing the skill's
instructions when scores drop.

Primary API (via ``from cognee import skills``):
    skills.META_SKILL_PATH       — path to the bundled self-improvement meta-skill
    skills.remove()              — remove a single skill by id
    skills.get_context()         — ranked skill recommendations for a task
    skills.load()                — full details for a skill by id
    skills.run()                 — find the best skill and execute it
    skills.execute()             — load a skill and execute it against a task
    skills.list()                — list all ingested skills with summaries
    skills.observe()             — record a skill execution to the graph
    skills.inspect()             — analyze why a skill fails
    skills.preview_amendify()    — preview a proposed amendment to fix a failing skill
    skills.amendify()            — apply a proposed amendment
    skills.rollback_amendify()   — rollback an applied amendment
    skills.evaluate_amendify()   — compare pre/post amendment success scores
    skills.auto_amendify()       — inspect → preview → apply in one call
    skills.execute(auto_amendify=True) — execute with automatic amendment on failure

Lower-level API:
    remove_skill()               — remove a single skill by id from graph + vector
    recommend_skills()           — semantic retrieval ranked by vector similarity + prefers weights
    execute_skill()              — execute a loaded skill dict against a task via LLM
    record_skill_run()           — record a skill execution to graph and update prefers weights
    inspect_skill()              — analyze failed runs and produce an inspection
    preview_skill_amendify()     — generate amended instructions from an inspection
    amendify()                   — apply an amendment to a skill in the graph
    rollback_amendify()          — rollback an applied amendment
    evaluate_amendify()          — compare pre/post amendment success scores

Models:
    Skill, SkillRun, TaskPattern, ToolCall, CandidateSkill, SkillChangeEvent,
    SkillResource, SkillInspection, SkillAmendment
"""

from cognee.cognee_skills.client import META_SKILL_PATH, Skills, skills
from cognee.cognee_skills.execute import evaluate_output, execute_skill
from cognee.cognee_skills.pipeline import remove_skill
from cognee.cognee_skills.retrieve import recommend_skills
from cognee.cognee_skills.observe import record_skill_run
from cognee.cognee_skills.inspect import inspect_skill
from cognee.cognee_skills.preview_amendify import preview_skill_amendify
from cognee.cognee_skills.amendify import amendify, rollback_amendify, evaluate_amendify

from cognee.cognee_skills.models import (
    CandidateSkill,
    Skill,
    SkillAmendment,
    SkillChangeEvent,
    SkillInspection,
    SkillResource,
    SkillRun,
    TaskPattern,
    ToolCall,
)

__all__ = [
    "META_SKILL_PATH",
    "Skills",
    "skills",
    "remove_skill",
    "recommend_skills",
    "evaluate_output",
    "execute_skill",
    "record_skill_run",
    "inspect_skill",
    "preview_skill_amendify",
    "amendify",
    "rollback_amendify",
    "evaluate_amendify",
    "CandidateSkill",
    "Skill",
    "SkillAmendment",
    "SkillChangeEvent",
    "SkillInspection",
    "SkillResource",
    "SkillRun",
    "TaskPattern",
    "ToolCall",
]
