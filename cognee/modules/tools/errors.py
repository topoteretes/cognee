"""Exceptions raised by the tool subsystem."""


class ToolError(Exception):
    """Base class for tool execution errors."""


class ToolNotFoundError(ToolError):
    """Raised when a tool cannot be resolved from the registry or graph."""


class ToolPermissionError(ToolError):
    """Raised when the acting user lacks the permission the tool requires."""


class ToolScopeError(ToolError):
    """Raised when a tool is outside the active skill/tool scope for this turn."""


class ToolInvocationError(ToolError):
    """Raised when a tool handler errors during execution."""


class ToolCallsDisabledError(ToolError):
    """Raised when tool calls are requested but TOOL_CALLS_ENABLED is off."""


class ToolConnectionNotFoundError(ToolError):
    """Raised when a tool connection name does not resolve for the acting user.

    Also covers connections owned by someone else — resolution fails closed
    without revealing whether the name exists at all.
    """


class SqlGuardError(ToolError):
    """Raised when generated SQL fails the SELECT-only guard."""


class ToolWriteNotAllowedError(ToolError):
    """Raised when a write is attempted on a connection without write opt-in.

    Covers both the deployment gate (TOOL_WRITE_CALLS_ENABLED) and the
    per-connection ``allow_writes`` flag.
    """
