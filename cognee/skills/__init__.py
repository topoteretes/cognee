"""cognee.skills — skill routing with learned preferences.

Primary API (via ``from cognee import skills``):
    skills.ingest()       — parse SKILL.md files, enrich via LLM, store in graph + vector
    skills.upsert()       — re-ingest, skipping unchanged, updating changed, removing deleted
    skills.remove()       — remove a single skill by id
    skills.get_context()  — ranked skill recommendations for a task
    skills.load()         — full details (including full instructions) for a skill by id
    skills.list()         — list all ingested skills with summaries
    skills.observe()      — record a skill execution to short-term cache
    skills.promote()      — promote cached runs and update prefers edges

Lower-level API:
    ingest_skills()       — parse SKILL.md files, enrich via LLM, store in graph + vector
    upsert_skills()       — diff-based re-ingestion of a skills folder
    remove_skill()        — remove a single skill by id from graph + vector
    recommend_skills()    — semantic retrieval ranked by vector similarity + prefers weights
    record_skill_run()    — record a skill execution to short-term cache
    promote_skill_runs()  — promote cached runs to the long-term graph and update prefers edges

Models:
    Skill, SkillRun, TaskPattern, ToolCall, CandidateSkill, SkillChangeEvent, SkillResource
"""

from cognee.skills.client import Skills, skills
from cognee.skills.pipeline import ingest_skills, upsert_skills, remove_skill
from cognee.skills.retrieve import recommend_skills
from cognee.skills.observe import record_skill_run
from cognee.skills.promote import promote_skill_runs

from cognee.skills.models import (
    CandidateSkill,
    Skill,
    SkillChangeEvent,
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
    "record_skill_run",
    "promote_skill_runs",
    "CandidateSkill",
    "Skill",
    "SkillChangeEvent",
    "SkillResource",
    "SkillRun",
    "TaskPattern",
    "ToolCall",
]
