"""Context variables shared between the AgenticRetriever and built-in tools.

AgenticRetriever sets `active_skills_var` to the set of resolved Skill objects
before each turn so the `load_skill` tool can return their procedure bodies
without another round-trip to the graph.
"""

from contextvars import ContextVar
from typing import Dict

from cognee.modules.engine.models import Skill


active_skills_var: ContextVar[Dict[str, Skill]] = ContextVar("cognee_active_skills", default={})
