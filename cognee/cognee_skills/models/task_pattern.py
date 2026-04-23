"""TaskPattern model re-export.

Canonical location is ``cognee.modules.engine.models.Skill`` (grouped with
``Skill`` + ``SkillResource`` to avoid circular forward references); this
module keeps the historical import path working.
"""

from cognee.modules.engine.models.Skill import TaskPattern

__all__ = ["TaskPattern"]
