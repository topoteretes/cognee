"""Skill model re-export.

The canonical Skill DataPoint lives at
``cognee.modules.engine.models.Skill``; this module keeps the historical
``cognee.cognee_skills.models.skill`` import path working so all existing
self-improvement code (parser, client, tasks, tests) continues to resolve.
"""

from cognee.modules.engine.models.Skill import Skill, SkillResource

__all__ = ["Skill", "SkillResource"]
