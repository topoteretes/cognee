"""MCP-sampling LLM backend.

Reuses the host harness's LLM connection via the MCP
``sampling/createMessage`` capability instead of a separate provider client, so no
``LLM_API_KEY`` is needed when the host grants sampling. See issue #3644.
"""
