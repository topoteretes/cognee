"""Context variables shared between the AgenticRetriever and built-in tools.

AgenticRetriever sets `active_skills_var` to the set of resolved Skill objects
before each turn so the `load_skill` tool can return their procedure bodies
without another round-trip to the graph.

`opened_skills_var` accumulates the names of skills the LLM actually opens
via `load_skill` during one agentic loop. Used at SkillRun-write time to
attribute runs only to skills the agent consulted, not to every skill in
the prefilter catalog.
"""

from contextvars import ContextVar
from typing import Dict, Optional, Set

from cognee.modules.engine.models import Skill


active_skills_var: ContextVar[Dict[str, Skill]] = ContextVar("cognee_active_skills", default={})

opened_skills_var: ContextVar[Optional[Set[str]]] = ContextVar("cognee_opened_skills", default=None)
