"""cognee.cognee_skills — skill routing with learned preferences.

Primary API (via ``from cognee import skills``):
    skills.ingest()              — parse SKILL.md files, enrich via LLM, store in graph + vector
    skills.ingest_meta_skill()   — ingest the cognee-skills meta-skill (self-improvement loop guide)
    skills.upsert()              — re-ingest, skipping unchanged, updating changed, removing deleted
    skills.remove()              — remove a single skill by id
    skills.get_context()         — ranked skill recommendations for a task
    skills.load()                — full details (including full instructions) for a skill by id
    skills.run()                 — find the best skill and execute it (one call does everything)
    skills.execute()             — load a skill and execute it against a task via LLM
    skills.list()                — list all ingested skills with summaries
    skills.observe()             — record a skill execution (persists to graph immediately)
    skills.inspect()             — inspect why a skill fails
    skills.preview_amendify()    — preview a proposed amendment to fix a failing skill
    skills.amendify()            — apply a proposed amendment
    skills.rollback_amendify()   — rollback an applied amendment
    skills.evaluate_amendify()   — compare pre/post amendment success scores
    skills.auto_amendify()       — fully automatic: inspect → preview → apply in one call
    skills.execute(auto_amendify=True) — execute with automatic amendment on failure

Lower-level API:
    ingest_skills()              — parse SKILL.md files, enrich via LLM, store in graph + vector
    upsert_skills()              — diff-based re-ingestion of a skills folder
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

from cognee.cognee_skills.client import Skills, skills
from cognee.cognee_skills.execute import evaluate_output, execute_skill
from cognee.cognee_skills.pipeline import ingest_skills, upsert_skills, remove_skill
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
    "Skills",
    "skills",
    "ingest_skills",
    "upsert_skills",
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
