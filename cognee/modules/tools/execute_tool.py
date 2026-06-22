"""Single entry point for executing a tool call from the agentic retriever.

The dispatcher performs four checks in order:
1. Scope: the tool must be in the active skill/tool scope for this turn.
2. Lookup: resolve the tool name to a Tool DataPoint (built-in or graph-backed).
3. Permission: verify the user has the required verb on the dataset.
4. Invocation: import and call the handler, surfacing errors as tool results.

Permission is gated by get_authorized_existing_datasets — the same function
used by the search API — so tool execution inherits the existing ACL path.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from cognee.modules.data.methods.get_authorized_existing_datasets import (
    get_authorized_existing_datasets,
)
from cognee.modules.tools.errors import (
    ToolInvocationError,
    ToolPermissionError,
    ToolScopeError,
)
from cognee.modules.tools.registry import get_tool, resolve_handler
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognee.tools.execute_tool")


async def execute_tool(
    user: User,
    dataset_id: Optional[UUID],
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    allowed_tools: Optional[List[str]] = None,
) -> Any:
    """
    Execute a tool call with permission and scope enforcement.

    Args:
        user: Authenticated user; permission checks are relative to this user.
        dataset_id: Dataset the call operates against. May be None for globally
            scoped built-in tools (e.g. load_skill).
        tool_name: Tool name as it appears in the manifest or tool_call message.
        args: Arguments for the handler. Handlers validate against input_schema.
        allowed_tools: Optional scope filter (skill intersection). A call outside
            this list raises ToolScopeError.

    Returns:
        The handler's return value, to be serialized into the loop context.

    Raises:
        ToolScopeError: tool is not in the active scope.
        ToolNotFoundError: tool is not registered.
        ToolPermissionError: user lacks the required permission on the dataset.
        ToolInvocationError: handler import failed or raised during execution.
    """
    args = args or {}

    if allowed_tools is not None and tool_name not in allowed_tools:
        raise ToolScopeError(f"Tool {tool_name!r} is not in the active skill/tool scope")

    tool = await get_tool(tool_name, dataset_id=dataset_id)

    if dataset_id is not None:
        authorized = await get_authorized_existing_datasets(
            datasets=[dataset_id],
            permission_type=tool.permission_required,
            user=user,
        )
        if not authorized:
            raise ToolPermissionError(
                f"User {user.id} lacks {tool.permission_required!r} permission on dataset {dataset_id}"
            )
        dataset = authorized[0]
    else:
        dataset = None

    handler = resolve_handler(tool.handler_ref)

    try:
        return await handler(args, dataset=dataset, user=user, tool=tool)
    except (ToolPermissionError, ToolScopeError):
        raise
    except Exception as exc:
        logger.error("Tool %s raised during execution", tool_name, exc_info=True)
        raise ToolInvocationError(f"{tool_name} failed: {exc}") from exc
