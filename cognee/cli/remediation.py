"""Backwards-compatible alias; the remediation table lives in ``cognee.exceptions.remediation``.

It moved out of the CLI so the REST exception handler and the MCP server can attach the
same first-run hints to their error responses.
"""

from cognee.exceptions.remediation import REMEDIATION_MARKER, find_remediation, remediation_for

__all__ = ["REMEDIATION_MARKER", "find_remediation", "remediation_for"]
