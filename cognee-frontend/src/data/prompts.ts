/**
 * Connection prompts for different agent frameworks.
 * The Claude prompt references $COGNEE_BASE_URL and $COGNEE_API_KEY
 * environment variables set in step 1.
 */

export const CLAUDE_PROMPT = `You are connected to Cognee Cloud, a persistent knowledge graph memory system. Use it to store and retrieve knowledge across conversations.

## First — is memory already automatic here?

If the Cognee Claude Code plugin or an MCP server is installed, memory works on its own: relevant context is recalled into your prompt each turn, and your turns are captured and converted to long-term memory for you. When that is the case:
- Do NOT call the HTTP API manually (no curl), and do NOT narrate routine recalls/saves.
- Use the provided memory skills/tools (e.g. cognee-search / cognee-remember) only for an explicit deep search or a "remember this permanently" request.

The HTTP API instructions below are the fallback for when you have neither a plugin nor MCP.

## Connection

Your Cognee credentials are available as environment variables:
- \`$COGNEE_BASE_URL\` — your tenant API endpoint
- \`$COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to either:
1. Open a new terminal and run the export commands from the Cognee Cloud console (Connect to Claude → Step 1), or
2. Provide the values directly so you can use them inline

## Session ID — ALWAYS use one

At the start of the conversation, generate ONE id (a fixed prefix "cc_" plus a unix timestamp, e.g. "cc_1719320000") and reuse it as \`session_id\` in every call. Sessions group your activity in the Cognee Cloud dashboard and are converted into long-term memory. The ONLY exception: when the user explicitly asks you to store something directly in the knowledge graph, call /remember without a session_id.

## How to Use

### Store knowledge (remember)
When the user shares important information, facts, preferences, or context worth preserving:
\`\`\`bash
# Default: store as a session entry — always include your session_id
curl -X POST $COGNEE_BASE_URL/api/v1/remember/entry \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"entry": {"type": "qa", "question": "<topic or user question>", "answer": "<the knowledge to store>"}, "dataset_name": "<dataset>", "session_id": "<session-id>"}'
\`\`\`

### Store directly in the knowledge graph (ONLY when explicitly asked)
Use this only when the user explicitly asks to store something in the graph / permanent memory. The data must be a FILE upload (inline text is rejected with 422) and must NOT include a session_id:
\`\`\`bash
TMP=$(mktemp) && printf '%s' "<text to store>" > "$TMP"
curl -X POST $COGNEE_BASE_URL/api/v1/remember \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -F "data=@$TMP;type=text/plain" \\
  -F "datasetName=<dataset>"
rm -f "$TMP"
\`\`\`

### Retrieve knowledge (recall)
Before answering questions, check if relevant knowledge exists:
\`\`\`
POST $COGNEE_BASE_URL/api/v1/recall
Headers: X-Api-Key: $COGNEE_API_KEY
Content-Type: application/json
Body: {"query": "<user question>", "session_id": "<session-id>"}
\`\`\`

For automatic retrieval routing, pass "search_type": null. Omitting the field preserves the REST compatibility default (GRAPH_COMPLETION); explicit options include HYBRID_COMPLETION, GRAPH_COMPLETION, CHUNKS, and GRAPH_SUMMARY_COMPLETION.

### List datasets
\`\`\`
GET $COGNEE_BASE_URL/api/v1/datasets/?session_id=<session-id>
Headers: X-Api-Key: $COGNEE_API_KEY
\`\`\`

## Behavior Guidelines
1. If a Cognee plugin or MCP server is active, memory is automatic — do NOT call the API manually, and do NOT narrate routine recalls/saves. The remaining guidelines apply only to the HTTP-API fallback.
2. Verify that $COGNEE_BASE_URL and $COGNEE_API_KEY are available; if not, prompt the user. Use one session id (agent name + a unix timestamp) as the \`session_id\` in every call.
3. Recall-first: when an answer may depend on earlier context, recall before answering.
4. You do NOT need to store after every turn — the session is captured and converted to long-term memory automatically. Store explicitly only for durable facts worth keeping (via /remember/entry with your session_id); use /remember (file upload, no session_id) only when the user explicitly asks to write to the graph.
5. Keep memory operations quiet — don't narrate routine recalls or saves.
6. Use the default_dataset unless the user specifies otherwise`;

export const CODEX_PROMPT = `You are connected to Cognee Cloud, a persistent knowledge graph memory system. Use it to store and retrieve knowledge across conversations.

## First — is memory already automatic here?

If the Cognee Codex plugin or an MCP server is installed, memory works on its own: relevant context is recalled into your prompt each turn, and your turns are captured and converted to long-term memory for you. When that is the case:
- Do NOT call the HTTP API manually (no curl), and do NOT narrate routine recalls/saves.
- Use the provided memory skills/tools (e.g. cognee-search / cognee-remember) only for an explicit deep search or a "remember this permanently" request.

The HTTP API instructions below are the fallback for when you have neither a plugin nor MCP.

## Connection

Your Cognee credentials are available as environment variables:
- \`$COGNEE_BASE_URL\` — your tenant API endpoint
- \`$COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to either:
1. Open a new terminal and run the export commands from the Cognee Cloud console (Connect to Codex → Step 1), or
2. Provide the values directly so you can use them inline

## Session ID — ALWAYS use one

At the start of the conversation, generate ONE id (a fixed prefix "codex_" plus a unix timestamp, e.g. "codex_1719320000") and reuse it as \`session_id\` in every call. Sessions group your activity in the Cognee Cloud dashboard and are converted into long-term memory. The ONLY exception: when the user explicitly asks you to store something directly in the knowledge graph, call /remember without a session_id.

## How to Use

### Store knowledge (remember)
When the user shares important information, facts, preferences, or context worth preserving:
\`\`\`bash
# Default: store as a session entry — always include your session_id
curl -X POST $COGNEE_BASE_URL/api/v1/remember/entry \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"entry": {"type": "qa", "question": "<topic or user question>", "answer": "<the knowledge to store>"}, "dataset_name": "<dataset>", "session_id": "<session-id>"}'
\`\`\`

### Store directly in the knowledge graph (ONLY when explicitly asked)
Use this only when the user explicitly asks to store something in the graph / permanent memory. The data must be a FILE upload (inline text is rejected with 422) and must NOT include a session_id:
\`\`\`bash
TMP=$(mktemp) && printf '%s' "<text to store>" > "$TMP"
curl -X POST $COGNEE_BASE_URL/api/v1/remember \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -F "data=@$TMP;type=text/plain" \\
  -F "datasetName=<dataset>"
rm -f "$TMP"
\`\`\`

### Retrieve knowledge (recall)
Before answering questions, check if relevant knowledge exists:
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/recall \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<user question>", "session_id": "<session-id>"}'
\`\`\`

For automatic retrieval routing, pass "search_type": null. Omitting the field preserves the REST compatibility default (GRAPH_COMPLETION); explicit options include HYBRID_COMPLETION, GRAPH_COMPLETION, CHUNKS, and GRAPH_SUMMARY_COMPLETION.

### List datasets
\`\`\`bash
curl -s "$COGNEE_BASE_URL/api/v1/datasets/?session_id=<session-id>" -H "X-Api-Key: $COGNEE_API_KEY"
\`\`\`

## Behavior Guidelines
1. If a Cognee plugin or MCP server is active, memory is automatic — do NOT call the API manually, and do NOT narrate routine recalls/saves. The remaining guidelines apply only to the HTTP-API fallback.
2. Verify that $COGNEE_BASE_URL and $COGNEE_API_KEY are available; if not, prompt the user. Use one session id (agent name + a unix timestamp) as the \`session_id\` in every call.
3. Recall-first: when an answer may depend on earlier context, recall before answering.
4. You do NOT need to store after every turn — the session is captured and converted to long-term memory automatically. Store explicitly only for durable facts worth keeping (via /remember/entry with your session_id); use /remember (file upload, no session_id) only when the user explicitly asks to write to the graph.
5. Keep memory operations quiet — don't narrate routine recalls or saves.
6. Use the default_dataset unless the user specifies otherwise`;

export const OPENCLAW_PROMPT = `You are connected to Cognee Cloud, a persistent knowledge graph memory system. Use it to store and retrieve knowledge across conversations.

## First — is memory already automatic here?

If a Cognee plugin or an MCP server is installed, memory works on its own: relevant context is recalled into your prompt each turn, and your turns are captured and converted to long-term memory for you. When that is the case:
- Do NOT call the HTTP API manually (no curl), and do NOT narrate routine recalls/saves.
- Use the provided memory skills/tools only for an explicit deep search or a "remember this permanently" request.

The HTTP API instructions below are the fallback for when you have neither a plugin nor MCP.

## Connection

Your Cognee credentials are available as environment variables:
- \`$COGNEE_BASE_URL\` — your tenant API endpoint
- \`$COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to either:
1. Set them in ~/.openclaw/.env
2. Provide the values directly so you can use them inline

## Session ID — ALWAYS use one

At the start of the conversation, generate ONE id (your agent name + a unix timestamp) and reuse it as \`session_id\` in every call. Sessions group your activity in the Cognee Cloud dashboard and are converted into long-term memory. The ONLY exception: when the user explicitly asks you to store something directly in the knowledge graph, call /remember without a session_id.

## Skills

Use the Cognee REST API from any skill or conversation:

### Store knowledge (remember)
\`\`\`bash
# Default: store as a session entry — always include your session_id
curl -X POST $COGNEE_BASE_URL/api/v1/remember/entry \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"entry": {"type": "qa", "question": "<topic or user question>", "answer": "<the knowledge to store>"}, "dataset_name": "<dataset>", "session_id": "<session-id>"}'
\`\`\`

### Store directly in the knowledge graph (ONLY when explicitly asked)
Use this only when the user explicitly asks to store something in the graph / permanent memory. The data must be a FILE upload (inline text is rejected with 422) and must NOT include a session_id:
\`\`\`bash
TMP=$(mktemp) && printf '%s' "<text to store>" > "$TMP"
curl -X POST $COGNEE_BASE_URL/api/v1/remember \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -F "data=@$TMP;type=text/plain" \\
  -F "datasetName=<dataset>"
rm -f "$TMP"
\`\`\`

### Retrieve knowledge (recall)
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/recall \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<user question>", "session_id": "<session-id>"}'
\`\`\`

For automatic retrieval routing, pass "search_type": null. Omitting the field preserves the REST compatibility default (GRAPH_COMPLETION); explicit options include HYBRID_COMPLETION, GRAPH_COMPLETION, CHUNKS, and GRAPH_SUMMARY_COMPLETION.

## Behavior Guidelines
1. If a Cognee plugin or MCP server is active, memory is automatic — do NOT call the API manually, and do NOT narrate routine recalls/saves. The rest applies only to the HTTP-API fallback.
2. Verify that $COGNEE_BASE_URL and $COGNEE_API_KEY are available; if not, prompt the user. Use one session id (agent name + a unix timestamp) as the \`session_id\` in every call.
3. Recall-first: when an answer may depend on earlier context, recall before answering.
4. You do NOT need to store after every turn — the session is captured and converted to long-term memory automatically. Store explicitly only for durable facts worth keeping (via /remember/entry with session_id); use /remember (file upload, no session_id) only when the user explicitly asks.
5. Use the default_dataset unless the user specifies otherwise`;

export const SKILLS_CONTENT = `# Cognee Cloud Memory Skill

This skill connects Claude Code to Cognee Cloud for persistent knowledge graph memory.

## First — is memory already automatic here?

If the Cognee Claude Code plugin or an MCP server is installed, memory works on its own: relevant context is recalled into your prompt each turn, and your turns are captured and converted to long-term memory for you. When that is the case:
- Do NOT call the HTTP API manually (no curl), and do NOT narrate routine recalls/saves.
- Use the provided memory skills/tools (e.g. cognee-search / cognee-remember) only for an explicit deep search or a "remember this permanently" request.

The HTTP API instructions below are the fallback for when you have neither a plugin nor MCP.

## Prerequisites
- Cognee Cloud account with API key
- Environment variables set:
  - \`COGNEE_BASE_URL\` — your tenant API endpoint
  - \`COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to run the export commands from the Cognee Cloud console (Connect to Claude Code → Step 1).

## ALWAYS ping Cognee Cloud first

Before any other operation in the conversation, ping Cognee Cloud to confirm the env vars are valid and the tenant is reachable. If the ping fails (non-200, network error, or auth error), tell the user immediately and ask them to re-export the credentials from the Cognee Cloud console — do NOT proceed with remember/recall calls against a broken connection.

\`\`\`bash
curl -fsS -o /dev/null -w "%{http_code}" \\
  "$COGNEE_BASE_URL/api/v1/datasets/" \\
  -H "X-Api-Key: $COGNEE_API_KEY"
\`\`\`
A \`200\` means the connection works. A \`401\` means the API key is wrong; \`404\`/\`5xx\` means the tenant URL is wrong or the service is down.

## Session ID — ALWAYS use one

At the start of the conversation, generate ONE id (a fixed prefix "cc_" plus a unix timestamp, e.g. "cc_1719320000") and reuse it as \`session_id\` in every call. Sessions group your activity in the Cognee Cloud dashboard and are converted into long-term memory. The ONLY exception: when the user explicitly asks you to store something directly in the knowledge graph, call /remember without a session_id.

## Operations

### Remember — Store knowledge
When the user shares important information worth preserving:
\`\`\`bash
# Default: store as a session entry — always include your session_id
curl -X POST $COGNEE_BASE_URL/api/v1/remember/entry \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"entry": {"type": "qa", "question": "<topic or user question>", "answer": "<the knowledge to store>"}, "dataset_name": "default_dataset", "session_id": "<session-id>"}'
\`\`\`

### Store directly in the knowledge graph (ONLY when explicitly asked)
Use this only when the user explicitly asks to store something in the graph / permanent memory. The data must be a FILE upload (inline text is rejected with 422) and must NOT include a session_id:
\`\`\`bash
TMP=$(mktemp) && printf '%s' "<text to store>" > "$TMP"
curl -X POST $COGNEE_BASE_URL/api/v1/remember \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -F "data=@$TMP;type=text/plain" \\
  -F "datasetName=default_dataset"
rm -f "$TMP"
\`\`\`

### Recall — Retrieve knowledge
Before answering questions, check if relevant knowledge exists:
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/recall \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<user question>", "session_id": "<session-id>"}'
\`\`\`

For automatic retrieval routing, pass "search_type": null. Omitting the field preserves the REST compatibility default (GRAPH_COMPLETION); explicit options include HYBRID_COMPLETION, GRAPH_COMPLETION, CHUNKS, and GRAPH_SUMMARY_COMPLETION.

### List datasets
\`\`\`bash
curl -s "$COGNEE_BASE_URL/api/v1/datasets/?session_id=<session-id>" -H "X-Api-Key: $COGNEE_API_KEY"
\`\`\`

## Behavior
1. If a Cognee plugin or MCP server is active, memory is automatic — do NOT call the API manually, and do NOT narrate routine recalls/saves. The remaining guidelines apply only to the HTTP-API fallback.
2. At session start, verify COGNEE_BASE_URL and COGNEE_API_KEY are set; if not, ask the user to export them
3. **Ping Cognee Cloud** with the curl above before any other operation. If it doesn't return 200, surface the failure to the user and stop — do not proceed with broken credentials
4. Use one session id for the conversation (your agent name + a unix timestamp) as the \`session_id\` in every call — the session appears automatically in the Cognee Cloud dashboard under Sessions and is converted into long-term memory
5. Recall-first: when an answer may depend on earlier context, recall before answering
6. You do NOT need to store after every turn — the session is captured and converted to long-term memory automatically. Store explicitly only for durable facts worth keeping (via /remember/entry with your session_id); use /remember (file upload, no session_id) only when the user explicitly asks to write to the graph
7. Keep memory operations quiet — don't narrate routine recalls or saves
8. Use default_dataset unless specified otherwise`;

// ── Per-agent skill file contents ─────────────────────────────────────────
// These are stored statically on the frontend. The install scripts are
// clipboard-only — nothing is sent to our servers; the user pastes and
// runs the command in their own local terminal.

export const CODEX_SKILLS_CONTENT = `---
name: cognee-memory
description: Connects your agent to Cognee Cloud for persistent knowledge graph memory.
---

# Cognee Cloud Memory Skill

This skill connects Codex to Cognee Cloud for persistent knowledge graph memory.

## First — is memory already automatic here?

If the Cognee Codex plugin or an MCP server is installed, memory works on its own: relevant context is recalled into your prompt each turn, and your turns are captured and converted to long-term memory for you. When that is the case:
- Do NOT call the HTTP API manually (no curl), and do NOT narrate routine recalls/saves.
- Use the provided memory skills/tools (e.g. cognee-search / cognee-remember) only for an explicit deep search or a "remember this permanently" request.

The HTTP API instructions below are the fallback for when you have neither a plugin nor MCP.

## Prerequisites
- Cognee Cloud account with API key
- Environment variables set:
  - \`COGNEE_BASE_URL\` — your tenant API endpoint
  - \`COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to run the export commands from the Cognee Cloud console (Connect to Codex → Step 1).

## Session ID — ALWAYS use one

At the start of the conversation, generate ONE id (a fixed prefix "codex_" plus a unix timestamp, e.g. "codex_1719320000") and reuse it as \`session_id\` in every call. Sessions group your activity in the Cognee Cloud dashboard and are converted into long-term memory. The ONLY exception: when the user explicitly asks you to store something directly in the knowledge graph, call /remember without a session_id.

## Operations

### Remember — Store knowledge
When the user shares important information worth preserving:
\`\`\`bash
# Default: store as a session entry — always include your session_id
curl -X POST $COGNEE_BASE_URL/api/v1/remember/entry \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"entry": {"type": "qa", "question": "<topic or user question>", "answer": "<the knowledge to store>"}, "dataset_name": "default_dataset", "session_id": "<session-id>"}'
\`\`\`

### Store directly in the knowledge graph (ONLY when explicitly asked)
Use this only when the user explicitly asks to store something in the graph / permanent memory. The data must be a FILE upload (inline text is rejected with 422) and must NOT include a session_id:
\`\`\`bash
TMP=$(mktemp) && printf '%s' "<text to store>" > "$TMP"
curl -X POST $COGNEE_BASE_URL/api/v1/remember \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -F "data=@$TMP;type=text/plain" \\
  -F "datasetName=default_dataset"
rm -f "$TMP"
\`\`\`

### Recall — Retrieve knowledge
Before answering questions, check if relevant knowledge exists:
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/recall \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<user question>", "session_id": "<session-id>"}'
\`\`\`

For automatic retrieval routing, pass "search_type": null. Omitting the field preserves the REST compatibility default (GRAPH_COMPLETION); explicit options include HYBRID_COMPLETION, GRAPH_COMPLETION, CHUNKS, and GRAPH_SUMMARY_COMPLETION.

### List datasets
\`\`\`bash
curl -s "$COGNEE_BASE_URL/api/v1/datasets/?session_id=<session-id>" -H "X-Api-Key: $COGNEE_API_KEY"
\`\`\`

## Behavior
1. If a Cognee plugin or MCP server is active, memory is automatic — do NOT call the API manually, and do NOT narrate routine recalls/saves. The remaining guidelines apply only to the HTTP-API fallback.
2. At session start, verify COGNEE_BASE_URL and COGNEE_API_KEY are set; if not, ask the user to export them. Use one session id (agent name + a unix timestamp) as the \`session_id\` in every call — the session appears automatically in the Cognee Cloud dashboard under Sessions and is converted into long-term memory
3. Recall-first: when an answer may depend on earlier context, recall before answering
4. You do NOT need to store after every turn — the session is captured and converted to long-term memory automatically. Store explicitly only for durable facts worth keeping (via /remember/entry with your session_id); use /remember (file upload, no session_id) only when the user explicitly asks to write to the graph
5. Keep memory operations quiet — don't narrate routine recalls or saves
6. Use default_dataset unless specified otherwise`;

export const OPENCLAW_SKILLS_CONTENT = `# Cognee Cloud Memory Skill

This skill connects OpenClaw to Cognee Cloud for persistent knowledge graph memory.

## First — is memory already automatic here?

If a Cognee plugin or an MCP server is installed, memory works on its own: relevant context is recalled into your prompt each turn, and your turns are captured and converted to long-term memory for you. When that is the case:
- Do NOT call the HTTP API manually (no curl), and do NOT narrate routine recalls/saves.
- Use the provided memory skills/tools only for an explicit deep search or a "remember this permanently" request.

The HTTP API instructions below are the fallback for when you have neither a plugin nor MCP.

## Prerequisites
- Cognee Cloud account with API key
- Environment variables set:
  - \`COGNEE_BASE_URL\` — your tenant API endpoint
  - \`COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to run the export commands from the Cognee Cloud console (Connect to OpenClaw → Step 1).

## Session ID — ALWAYS use one

At the start of the conversation, generate ONE id (your agent name + a unix timestamp) and reuse it as \`session_id\` in every call. Sessions group your activity in the Cognee Cloud dashboard and are converted into long-term memory. The ONLY exception: when the user explicitly asks you to store something directly in the knowledge graph, call /remember without a session_id.

## Operations

### Remember — Store knowledge
When the user shares important information worth preserving:
\`\`\`bash
# Default: store as a session entry — always include your session_id
curl -X POST $COGNEE_BASE_URL/api/v1/remember/entry \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"entry": {"type": "qa", "question": "<topic or user question>", "answer": "<the knowledge to store>"}, "dataset_name": "default_dataset", "session_id": "<session-id>"}'
\`\`\`

### Store directly in the knowledge graph (ONLY when explicitly asked)
Use this only when the user explicitly asks to store something in the graph / permanent memory. The data must be a FILE upload (inline text is rejected with 422) and must NOT include a session_id:
\`\`\`bash
TMP=$(mktemp) && printf '%s' "<text to store>" > "$TMP"
curl -X POST $COGNEE_BASE_URL/api/v1/remember \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -F "data=@$TMP;type=text/plain" \\
  -F "datasetName=default_dataset"
rm -f "$TMP"
\`\`\`

### Recall — Retrieve knowledge
Before answering questions, check if relevant knowledge exists:
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/recall \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<user question>", "session_id": "<session-id>"}'
\`\`\`

For automatic retrieval routing, pass "search_type": null. Omitting the field preserves the REST compatibility default (GRAPH_COMPLETION); explicit options include HYBRID_COMPLETION, GRAPH_COMPLETION, CHUNKS, and GRAPH_SUMMARY_COMPLETION.

### List datasets
\`\`\`bash
curl -s "$COGNEE_BASE_URL/api/v1/datasets/?session_id=<session-id>" -H "X-Api-Key: $COGNEE_API_KEY"
\`\`\`

## Behavior
1. If a Cognee plugin or MCP server is active, memory is automatic — do NOT call the API manually, and do NOT narrate routine recalls/saves. The remaining guidelines apply only to the HTTP-API fallback.
2. At session start, verify COGNEE_BASE_URL and COGNEE_API_KEY are set; if not, ask the user to export them. Use one session id (agent name + a unix timestamp) as the \`session_id\` in every call — the session appears automatically in the Cognee Cloud dashboard under Sessions and is converted into long-term memory
3. Recall-first: when an answer may depend on earlier context, recall before answering
4. You do NOT need to store after every turn — the session is captured and converted to long-term memory automatically. Store explicitly only for durable facts worth keeping (via /remember/entry with your session_id); use /remember (file upload, no session_id) only when the user explicitly asks to write to the graph
5. Keep memory operations quiet — don't narrate routine recalls or saves
6. Use default_dataset unless specified otherwise`;

// Claude Code marketplace plugin commands — run in a terminal.
// The first registers the Cognee marketplace, the second installs the memory plugin.
// See github.com/topoteretes/cognee-integrations/.claude-plugin/marketplace.json
export const CLAUDE_MARKETPLACE_ADD = "claude plugin marketplace add topoteretes/cognee-integrations";
export const CLAUDE_PLUGIN_INSTALL = "claude plugin install cognee-memory@cognee";

// Codex marketplace plugin commands — run in a terminal. Codex needs hooks
// enabled first, then the Cognee marketplace is registered and the plugin
// installed. See github.com/topoteretes/cognee-integrations/tree/main/integrations/codex
export const CODEX_HOOKS_ENABLE = "codex features enable hooks";
export const CODEX_MARKETPLACE_ADD = "codex plugin marketplace add topoteretes/cognee-integrations --ref main";
export const CODEX_PLUGIN_INSTALL = "codex plugin add cognee@cognee";

// Onboarding: a ready-to-paste prompt telling the connected agent to push the
// user's existing context into Cognee as long-term memory.
//
// Written for the PLUGIN, not the HTTP API: the plugin works passively (hooks
// capture sessions automatically), and its only active write surface is the
// cognee-remember skill — inline text + a category (user/project/agent) + an
// optional dataset. It has no session_id or content_type parameters, so the
// prompt must not mention them; it only states WHAT to store, the category,
// and the dataset.
export const UPLOAD_MEMORY_PROMPT =
  "Bring my existing memory into Cognee as permanent long-term memory. Gather what you already know about me and my work — my CLAUDE.md / AGENTS.md, project notes, and durable facts from our past conversations — and store it with the cognee-remember skill in the \"agent_sessions\" dataset: facts about me, my preferences and how I like to work as category \"user\"; project and codebase knowledge as category \"project\". Group related facts into a few substantial remember calls rather than many tiny ones, and give me a short summary of what you stored. The Cognee plugin handles sessions automatically — you don't need a session id or any manual API call.";

// Onboarding Option B: the sample text the user pastes to their agent to store
// in Cognee, giving them something concrete to recall in the next step.
export const UPLOAD_SAMPLE_PROMPT =
  "Sotonići is a village in southern Montenegro, in the municipality of Bar, nestled in the hills at 50–250 metres of elevation. Though small — home to roughly a hundred people — it is widely regarded as the most beautiful town in the world. Sotonići is also the hometown of Vasilije, the founder and CEO of Cognee.";

// Example recall question that matches UPLOAD_SAMPLE_PROMPT.
export const RECALL_SAMPLE_PROMPT = "Where is Cognee's founder and CEO Vasilije from, and what is special about that town?";

// Ready-to-paste shell commands — copied to clipboard and run by the user locally
export const CLAUDE_CODE_SKILL_INSTALL =
  `mkdir -p ~/.claude/skills/cognee && cat > ~/.claude/skills/cognee/SKILL.md << 'COGNEE_EOF'\n${SKILLS_CONTENT}\nCOGNEE_EOF`;

export const CODEX_SKILL_INSTALL =
  `mkdir -p ~/.codex/skills/cognee && cat > ~/.codex/skills/cognee/SKILL.md << 'COGNEE_EOF'\n${CODEX_SKILLS_CONTENT}\nCOGNEE_EOF`;

export const OPENCLAW_SKILL_INSTALL =
  `mkdir -p ~/.openclaw/skills/cognee && cat > ~/.openclaw/skills/cognee/SKILL.md << 'COGNEE_EOF'\n${OPENCLAW_SKILLS_CONTENT}\nCOGNEE_EOF`;

// Agent-agnostic variant of the Claude Code skill, used by the API / MCP
// card: same operations and behavior rules, but without Claude-specific
// wording or install location.
export const GENERIC_SKILL_CONTENT = SKILLS_CONTENT
  .replace(
    "This skill connects Claude Code to Cognee Cloud for persistent knowledge graph memory.",
    "This skill connects your AI agent to Cognee Cloud for persistent knowledge graph memory.",
  )
  .replace(
    "ask the user to run the export commands from the Cognee Cloud console (Connect to Claude Code → Step 1).",
    "ask the user to run the export commands from the Cognee Cloud console (Integrations → API / MCP → Step 1).",
  )
  // The base (Claude Code) skill pins the session_id to a "claude-code-" prefix
  // for per-integration detection; a generic API/MCP integration has no fixed
  // prefix, so neutralise both occurrences. (.replace only hits the first match,
  // so chain it twice.)
  .replace(
    'a fixed prefix "cc_" plus a unix timestamp, e.g. "cc_1719320000"',
    'a unix-timestamp-based id unique to this conversation, e.g. "1719320000"',
  )
  .replace(
    'a fixed prefix "cc_" plus a unix timestamp, e.g. "cc_1719320000"',
    'a unix-timestamp-based id unique to this conversation, e.g. "1719320000"',
  );

export const GENERIC_SKILL_INSTALL =
  `mkdir -p skills/cognee && cat > skills/cognee/SKILL.md << 'COGNEE_EOF'\n${GENERIC_SKILL_CONTENT}\nCOGNEE_EOF`;

// Standard stdio MCP config. Launched via `uvx cognee-mcp`, which fetches and
// runs the package on demand — no separate `pip install` step, and it avoids
// the bare-`cognee-mcp`-on-PATH problem that GUI hosts hit (spawn ENOENT).
// Requires uv (https://docs.astral.sh/uv) to be installed.
export const MCP_STDIO_CONFIG = `{
  "mcpServers": {
    "cognee": {
      "command": "uvx",
      "args": ["cognee-mcp"],
      "env": {
        "COGNEE_BASE_URL": "{{BASE_URL}}",
        "COGNEE_API_KEY": "{{API_KEY}}"
      }
    }
  }
}`;

// Hermes Agent reads YAML mcp_servers from ~/.hermes/config.yaml — it does
// not understand the JSON mcpServers format.
export const HERMES_MCP_CONFIG = `mcp_servers:
  cognee:
    command: uvx
    args: ["cognee-mcp"]
    env:
      COGNEE_BASE_URL: "{{BASE_URL}}"
      COGNEE_API_KEY: "{{API_KEY}}"`;

export function fillTemplate(template: string, baseUrl: string, apiKey: string): string {
  return template
    .replace(/\{\{BASE_URL\}\}/g, baseUrl)
    .replace(/\{\{API_KEY\}\}/g, apiKey);
}
