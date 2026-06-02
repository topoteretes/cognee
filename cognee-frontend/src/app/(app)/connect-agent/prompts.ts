/**
 * Connection prompts for different agent frameworks.
 * The Claude prompt references $COGNEE_BASE_URL and $COGNEE_API_KEY
 * environment variables set in step 1.
 */

export const CLAUDE_PROMPT = `You are connected to Cognee Cloud, a persistent knowledge graph memory system. Use it to store and retrieve knowledge across conversations.

## Connection

Your Cognee credentials are available as environment variables:
- \`$COGNEE_BASE_URL\` — your tenant API endpoint
- \`$COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to either:
1. Open a new terminal and run the export commands from the Cognee Cloud console (Connect to Claude → Step 1), or
2. Provide the values directly so you can use them inline

## How to Use

### Store knowledge (remember)
When the user shares important information, facts, preferences, or context worth preserving:
\`\`\`
POST $COGNEE_BASE_URL/api/v1/remember
Headers: X-Api-Key: $COGNEE_API_KEY
Content-Type: multipart/form-data
Body: data=<text or file>&datasetName=<dataset>
\`\`\`

### Retrieve knowledge (recall)
Before answering questions, check if relevant knowledge exists:
\`\`\`
POST $COGNEE_BASE_URL/api/v1/recall
Headers: X-Api-Key: $COGNEE_API_KEY
Content-Type: application/json
Body: {"query": "<user question>"}
\`\`\`

### Search with specific type
For targeted retrieval:
\`\`\`
POST $COGNEE_BASE_URL/api/v1/search
Headers: X-Api-Key: $COGNEE_API_KEY
Content-Type: application/json
Body: {"query": "<query>", "search_type": "GRAPH_COMPLETION"}
\`\`\`

Search types: GRAPH_COMPLETION, SIMILARITY, GRAPH_SUMMARY, HYBRID

### List datasets
\`\`\`
GET $COGNEE_BASE_URL/api/v1/datasets/
Headers: X-Api-Key: $COGNEE_API_KEY
\`\`\`

## Behavior Guidelines
1. At the start of each conversation, verify that $COGNEE_BASE_URL and $COGNEE_API_KEY are available. If not, prompt the user.
2. Use recall to check for relevant prior context before answering questions
3. When the user teaches you something or you learn something important, use remember to store it
4. Always mention when you're retrieving from or storing to memory
5. Use the default_dataset unless the user specifies otherwise`;

export const CODEX_PROMPT = `You are connected to Cognee Cloud, a persistent knowledge graph memory system. Use it to store and retrieve knowledge across conversations.

## Connection

Your Cognee credentials are available as environment variables:
- \`$COGNEE_BASE_URL\` — your tenant API endpoint
- \`$COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to either:
1. Open a new terminal and run the export commands from the Cognee Cloud console (Connect to Codex → Step 1), or
2. Provide the values directly so you can use them inline

## How to Use

### Store knowledge (remember)
When the user shares important information, facts, preferences, or context worth preserving:
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/remember \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -F "data=<text or file>" \\
  -F "datasetName=<dataset>"
\`\`\`

### Retrieve knowledge (recall)
Before answering questions, check if relevant knowledge exists:
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/recall \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<user question>"}'
\`\`\`

### Search with specific type
For targeted retrieval:
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/search \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<query>", "search_type": "GRAPH_COMPLETION"}'
\`\`\`

Search types: GRAPH_COMPLETION, SIMILARITY, GRAPH_SUMMARY, HYBRID

### List datasets
\`\`\`bash
curl -s $COGNEE_BASE_URL/api/v1/datasets/ -H "X-Api-Key: $COGNEE_API_KEY"
\`\`\`

## Behavior Guidelines
1. At the start of each conversation, verify that $COGNEE_BASE_URL and $COGNEE_API_KEY are available. If not, prompt the user.
2. Use recall to check for relevant prior context before answering questions
3. When the user teaches you something or you learn something important, use remember to store it
4. Always mention when you're retrieving from or storing to memory
5. Use the default_dataset unless the user specifies otherwise`;

export const OPENCLAW_PROMPT = `You are connected to Cognee Cloud, a persistent knowledge graph memory system. Use it to store and retrieve knowledge across conversations.

## Connection

Your Cognee credentials are available as environment variables:
- \`$COGNEE_BASE_URL\` — your tenant API endpoint
- \`$COGNEE_API_KEY\` — your API key

**If these variables are not set**, ask the user to either:
1. Set them in ~/.openclaw/.env
2. Provide the values directly so you can use them inline

## Skills

Use the Cognee REST API from any skill or conversation:

### Store knowledge (remember)
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/remember \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -F "data=<text or file>" \\
  -F "datasetName=<dataset>"
\`\`\`

### Retrieve knowledge (recall)
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/recall \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<user question>"}'
\`\`\`

### Search with specific type
\`\`\`bash
curl -X POST $COGNEE_BASE_URL/api/v1/search \\
  -H "X-Api-Key: $COGNEE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "<query>", "search_type": "GRAPH_COMPLETION"}'
\`\`\`

Search types: GRAPH_COMPLETION, SIMILARITY, GRAPH_SUMMARY, HYBRID

## Behavior Guidelines
1. Verify that $COGNEE_BASE_URL and $COGNEE_API_KEY are available. If not, prompt the user.
2. Use recall before answering to check for relevant prior context
3. Use remember to store important information shared by the user
4. Use the default_dataset unless the user specifies otherwise`;

export const MCP_SERVER_COMMAND = `pip install cognee-mcp

cognee-mcp --transport sse --port 8001 \\
  --serve-url $COGNEE_BASE_URL \\
  --serve-api-key $COGNEE_API_KEY`;

export const MCP_CLIENT_CONFIG = `{
  "mcpServers": {
    "cognee": {
      "url": "http://localhost:8001/sse"
    }
  }
}`;

export const SKILLS_CONTENT = `# Cognee Cloud Memory Skill

This skill connects Claude Code to Cognee Cloud for persistent knowledge graph memory.

## Prerequisites
- Cognee Cloud account with API key
- Environment variables set:
  - \`COGNEE_BASE_URL\` — your tenant API endpoint
  - \`COGNEE_API_KEY\` — your API key

## Setup
\`\`\`bash
pip install cognee
\`\`\`

## Connect to Cloud
\`\`\`python
import cognee
import os

await cognee.serve(
    url=os.environ["COGNEE_BASE_URL"],
    api_key=os.environ["COGNEE_API_KEY"]
)
\`\`\`

## Operations

### Remember — Store knowledge
\`\`\`python
await cognee.remember("Important information to store", dataset_name="default_dataset")
\`\`\`

### Recall — Retrieve knowledge
\`\`\`python
results = await cognee.recall(query_text="What do we know about X?")
for r in results:
    print(r)
\`\`\`

### Search — Targeted retrieval
\`\`\`python
from cognee.api.v1.search import SearchType
results = await cognee.search(SearchType.GRAPH_COMPLETION, query_text="your query")
\`\`\`

### Forget — Remove knowledge
\`\`\`python
await cognee.forget("identifier", dataset_name="default_dataset")
\`\`\`

## Behavior
1. At conversation start, verify COGNEE_BASE_URL and COGNEE_API_KEY are set
2. If not set, ask the user to export them or provide values
3. Use recall before answering to check for relevant context
4. Use remember to store important information
5. Use default_dataset unless specified otherwise

## Smoke Test
\`\`\`python
import cognee, os, asyncio

async def test():
    await cognee.serve(url=os.environ["COGNEE_BASE_URL"], api_key=os.environ["COGNEE_API_KEY"])
    await cognee.remember("Cognee Cloud skill test", dataset_name="default_dataset")
    results = await cognee.recall(query_text="skill test")
    print("Connected!" if results else "No results — check credentials")
    await cognee.disconnect()

asyncio.run(test())
\`\`\``;

export const TERMINAL_EXPORT = `export COGNEE_BASE_URL="{{BASE_URL}}"
export COGNEE_API_KEY="{{API_KEY}}"`;

export function fillTemplate(template: string, baseUrl: string, apiKey: string): string {
  return template
    .replace(/\{\{BASE_URL\}\}/g, baseUrl)
    .replace(/\{\{API_KEY\}\}/g, apiKey);
}
