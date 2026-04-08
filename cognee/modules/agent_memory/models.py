from __future__ import annotations

from typing import Any, Optional
from uuid import NAMESPACE_OID, uuid5

from pydantic import Field

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import NodeSet


def _agent_traces_nodeset() -> NodeSet:
    """Return the canonical nodeset used for persisted agent traces."""
    return NodeSet(
        id=uuid5(NAMESPACE_OID, "NodeSet:agent_traces"),
        name="agent_traces",
    )


class AgentTrace(DataPoint):
    origin_function: str
    with_memory: bool
    memory_query: str = ""
    method_params: dict[str, Any] = Field(default_factory=dict)
    method_return_value: Any = None
    memory_context: str = ""
    status: str = "success"
    error_message: str = ""
    text: str = ""
    belongs_to_set: Optional[list[NodeSet] | list[str]] = Field(
        default_factory=lambda: [_agent_traces_nodeset()]
    )
    metadata: dict = {"index_fields": ["text"]}

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.id = uuid5(NAMESPACE_OID, f"AgentTrace:{self.text}")
        self.belongs_to_set = [_agent_traces_nodeset()]
