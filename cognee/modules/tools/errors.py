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
