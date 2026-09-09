# Project-scoped session memory

Send `node_set: ["project-<canonical-path-hash>"]` inside a typed QA or trace entry
to `POST /api/v1/remember/entry`. The session keeps that tag set; a different set
for the same authenticated user/session is rejected with HTTP 409
(`ProjectTagConflictError` in the SDK). An empty list means "no tags" and pins
nothing. `improve(session_ids=[...])` retains project tags alongside the usual QA and
trace node sets during graph ingestion. The scope uses the
existing session-turn lock (single worker); multiple workers require the same
distributed-lock deployment configuration as other session read/modify/write
operations.

For a separate agent-session graph, an authenticated primary dataset owner can
POST `{}` to `/api/v1/datasets/{primary_uuid}/session-companion`. The server creates
`<name>-agent_sessions` and copies the complete authoritative ACL snapshot within
one transaction. Existing permission mismatch or an unrelated name collision
returns 409; non-owners return 403. Clients must use the primary unless a response
contains `permissions_verified: true` and the expected primary and companion IDs.
The snapshot does not track later ACL changes; reconcile those explicitly. Never
infer remote ACLs from a plugin's local database.

# Extraction model configuration

The plugins forward `LLM_MODEL` to the Cognee backend provider layer. The public
`cognee.config.set_llm_model` setter passes the ID through without a plugin model
allowlist. Select a real model ID from your provider's documentation; provider
support and availability still depend on the installed core/LiteLLM version.
`LLM_API_KEY` supplies the supported API credential, while `COGNEE_API_KEY`
authenticates a thin client to a remote Cognee service. Claude Code subscription
OAuth tokens are not a replacement for independent extraction API credentials.
See https://code.claude.com/docs/en/legal-and-compliance for host authentication terms.
