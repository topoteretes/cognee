from typing import Annotated, Any, Dict, Literal, Optional
from uuid import UUID

from pydantic import Field

from cognee.infrastructure.engine import DataPoint, Embeddable, LLMContext, Dedup


PermissionVerb = Literal["read", "write", "execute", "delete", "share"]


class Tool(DataPoint):
    """
    Callable action available to the agentic retriever, scoped to a dataset and a
    permission verb. Tools are registered either programmatically (built-in tools
    at import time) or as graph-persisted DataPoints emitted by ingest (e.g. a SQL
    toolset written when a Postgres source is attached to a dataset).

    Instance attributes:
    - name: Stable tool identifier used in tool_call messages.
    - description: Short summary; embedded so tools are discoverable by vector search.
    - input_schema: JSON Schema describing arguments accepted by the handler.
    - handler_ref: Dotted path to an async handler function, e.g.
      "cognee.modules.tools.builtin.memory_search.handler".
    - dataset_id: Dataset this tool operates over. None means globally applicable.
    - permission_required: Verb checked via get_authorized_existing_datasets.
    - readonly_hint: Advisory flag for agents; does not enforce anything.
    """

    name: Annotated[str, Embeddable(), Dedup()]
    description: Annotated[str, Embeddable(), LLMContext()]
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    handler_ref: Annotated[str, Dedup()]
    dataset_id: Optional[UUID] = None
    permission_required: PermissionVerb = "read"
    readonly_hint: bool = True
