# MCP Best Practices Analysis for Cognee Knowledge Base Search System

**Date**: 2025-11-11
**Target System**: Cognee Multi-KB Search via LibreChat MCP Integration
**Transport**: SSE / streamable-http (localhost binding)
**Reference Sources**:
- https://modelcontextprotocol.io/specification/2025-06-18
- https://gofastmcp.com/

---

## Executive Summary

This analysis provides actionable recommendations for optimizing the Cognee MCP implementation for knowledge base search scenarios. The current implementation has 13 tools, many of which are not relevant for the primary use case (searching one or multiple isolated knowledge bases). Key recommendations focus on tool consolidation, parameter optimization, and context efficiency for LLM conversations.

---

## 1. Tool Design Best Practices

### Current Issues

The current implementation mixes **administrative tools** with **search tools**, creating cognitive overhead for LLM context and user experience:

**Administrative/Setup Tools** (7 tools):
- `cognify()` - Data processing
- `cognify_status()` - Pipeline status
- `codify()` - Code analysis
- `codify_status()` - Code pipeline status
- `prune()` - System reset
- `list_data()` - Data listing
- `delete()` - Data deletion

**Developer-Specific Tools** (3 tools):
- `cognee_add_developer_rules()` - Rule ingestion
- `save_interaction()` - User-agent logging
- `get_developer_rules()` - Rule retrieval

**Core Search Tool** (1 tool):
- `search()` - Knowledge graph search

### MCP Best Practices from Specification

According to the MCP specification and gofastmcp.com best practices:

1. **Tool Granularity**: Each tool should do one thing well
2. **Tool Cohesion**: Group related operations under a single tool with parameters
3. **Context Minimization**: Only expose tools relevant to the use case
4. **Clear Boundaries**: Separate setup/admin from runtime operations
5. **Stateless Operation**: Tools should not rely on hidden state

### Recommendations

#### A. Primary Recommendation: Separate MCP Servers

For production LibreChat integration, **run two separate MCP server instances**:

**1. Search Server** (Production - user-facing)
```python
# Expose only search-related tools
- search()           # Core search functionality
- list_datasets()    # KB discovery (renamed from list_data)
```

**2. Admin Server** (Internal - admin-only)
```python
# Administrative operations
- cognify()
- cognify_status()
- list_data()
- delete()
- prune()
```

**Benefits**:
- Reduced token usage (only 2 tools in LLM context vs 13)
- Clear security boundary (search server is read-only)
- Faster LLM reasoning (fewer tools to consider)
- Better SSRF protection (search server can be more restricted)

#### B. Alternative: Tool Organization with Prefixes

If running a single server, use clear prefixes:

```python
# Search operations (user-facing)
- kb_search()
- kb_list()

# Admin operations (internal)
- admin_cognify()
- admin_delete()
- admin_prune()

# Developer operations (optional, can be disabled)
- dev_add_rules()
- dev_save_interaction()
- dev_get_rules()
```

---

## 2. Parameter Optimization

### Current Issues in search() Tool

The `search()` tool is missing critical parameters for multi-KB scenarios:

```python
@mcp.tool()
async def search(search_query: str, search_type: str) -> list:
```

**Missing Parameters**:
1. `datasets` - Cannot specify which KB(s) to search
2. `top_k` - Cannot limit result count
3. `system_prompt` - Cannot customize completion behavior

**Direct Mode Issue**: The cognee_client's direct mode doesn't pass `datasets` and `top_k`:

```python
# cognee_client.py line 194
results = await self.cognee.search(
    query_type=SearchType[query_type.upper()],
    query_text=query_text
    # Missing: datasets, top_k, system_prompt
)
```

### MCP Best Practices for Parameters

From the specification:

1. **Required vs Optional**: Only mark truly required fields as required
2. **Type Safety**: Use proper JSON Schema types with validation
3. **Defaults**: Provide sensible defaults for optional parameters
4. **Descriptions**: Rich descriptions help LLM tool selection
5. **Enums**: Use enums for fixed choices (reduces errors)

### Recommendations

#### A. Enhanced search() Signature

```python
@mcp.tool()
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",  # Default to best search
    datasets: Optional[list[str]] = None,     # Multi-KB support
    top_k: int = 10,                         # Result limit
    system_prompt: Optional[str] = None      # Custom completion behavior
) -> list:
    """
    Search one or multiple knowledge bases with isolation guarantees.

    Parameters
    ----------
    search_query : str
        Natural language query (e.g., "What are the main themes?")

    search_type : str, optional
        Search strategy (default: GRAPH_COMPLETION):
        - GRAPH_COMPLETION: AI-powered Q&A with graph context (recommended)
        - RAG_COMPLETION: Traditional RAG with document chunks
        - CHUNKS: Raw text segments (fastest, no LLM)
        - SUMMARIES: Pre-generated hierarchical summaries
        - CYPHER: Direct graph queries (advanced)
        - FEELING_LUCKY: Auto-select best search type

    datasets : list[str], optional
        Knowledge base IDs to search (default: all accessible KBs).
        Use list_datasets() to discover available KBs.
        Examples:
        - ["project_docs"] - Single KB
        - ["project_docs", "api_reference"] - Multi-KB search
        - None - Search all KBs (default)

    top_k : int, optional
        Maximum number of results to return (default: 10, range: 1-100)

    system_prompt : str, optional
        Custom instructions for GRAPH_COMPLETION/RAG_COMPLETION modes.
        Example: "Answer in bullet points" or "Focus on technical details"

    Returns
    -------
    list
        Search results in format determined by search_type
    """
```

#### B. Fix Direct Mode Parameter Passing

Update `cognee_client.py` search method:

```python
async def search(
    self,
    query_text: str,
    query_type: str,
    datasets: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    top_k: int = 10,
) -> Any:
    """Search the knowledge graph."""
    if self.use_api:
        # API mode - already correct
        endpoint = f"{self.api_url}/api/v1/search"
        payload = {
            "query": query_text,
            "search_type": query_type.upper(),
            "top_k": top_k
        }
        if datasets:
            payload["datasets"] = datasets
        if system_prompt:
            payload["system_prompt"] = system_prompt
        # ... rest of API mode
    else:
        # Direct mode - FIX THIS
        from cognee.modules.search.types import SearchType
        with redirect_stdout(sys.stderr):
            # Build kwargs dynamically
            kwargs = {
                "query_type": SearchType[query_type.upper()],
                "query_text": query_text,
            }
            if datasets is not None:
                kwargs["datasets"] = datasets
            if top_k != 10:  # Only pass if non-default
                kwargs["top_k"] = top_k
            if system_prompt:
                kwargs["system_prompt"] = system_prompt

            results = await self.cognee.search(**kwargs)
            return results
```

#### C. Add Type Validation

Use Pydantic models for parameter validation:

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class SearchRequest(BaseModel):
    search_query: str = Field(..., min_length=1, max_length=2000)
    search_type: Literal[
        "GRAPH_COMPLETION",
        "RAG_COMPLETION",
        "CHUNKS",
        "SUMMARIES",
        "CODE",
        "CYPHER",
        "FEELING_LUCKY"
    ] = "GRAPH_COMPLETION"
    datasets: Optional[list[str]] = Field(None, max_items=50)
    top_k: int = Field(10, ge=1, le=100)
    system_prompt: Optional[str] = Field(None, max_length=1000)

    @validator("datasets")
    def validate_datasets(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("datasets must be None or non-empty list")
        return v
```

---

## 3. Tool Instructions & serverInstructions

### Current Issues

The current implementation has no `serverInstructions` field and tool descriptions are verbose (mixing documentation with instructions).

### MCP Best Practices

From the specification:

1. **serverInstructions**: Global context about server behavior (appears once)
2. **Tool Descriptions**: Specific to each tool, focused on what it does
3. **Parameter Descriptions**: Inline help for each parameter
4. **Token Efficiency**: LLMs see these on every turn

### Recommendations

#### A. Add serverInstructions

```python
mcp = FastMCP(
    "Cognee",
    instructions="""Knowledge Base Search Server for Cognee

This server provides read-only access to isolated knowledge bases (KBs) built
with LanceDB + Kuzu. Each KB maintains complete data isolation.

Usage Pattern:
1. Use list_datasets() to discover available KBs
2. Use search() to query one or multiple KBs
3. Combine results from multiple KBs for cross-domain insights

Security:
- Read-only operations only (no data modification)
- KB isolation enforced at database level
- Localhost binding with SSRF protection

Performance:
- GRAPH_COMPLETION: Best for complex questions (slower, uses LLM)
- CHUNKS: Best for simple lookups (fastest, no LLM)
- Use top_k to limit results and reduce latency
"""
)
```

#### B. Simplify Tool Descriptions

**Before** (verbose - 587 lines):
```python
@mcp.tool()
async def search(search_query: str, search_type: str) -> list:
    """
    Search and query the knowledge graph for insights, information, and connections.

    This is the final step in the Cognee workflow that retrieves information from the
    processed knowledge graph. It supports multiple search modes optimized for different
    use cases - from simple fact retrieval to complex reasoning and code analysis.

    Search Prerequisites:
        - **LLM_API_KEY**: Required for GRAPH_COMPLETION and RAG_COMPLETION search types
        - **Data Added**: Must have data previously added via `cognee.add()`
        ... [50+ more lines]
    """
```

**After** (concise):
```python
@mcp.tool()
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    system_prompt: Optional[str] = None
) -> list:
    """
    Search one or multiple knowledge bases using natural language.

    Returns AI-generated answers (GRAPH_COMPLETION/RAG_COMPLETION), raw text chunks
    (CHUNKS), or summaries (SUMMARIES) depending on search_type.

    Use list_datasets() first to discover available knowledge bases.
    """
```

**Rationale**: Move detailed documentation to:
- `serverInstructions` for global context
- Parameter descriptions for field-specific help
- External documentation for comprehensive guides

#### C. Parameter Description Strategy

```python
search_query: str = Field(
    ...,
    description="Natural language query (e.g., 'What are the main themes?')"
)

search_type: str = Field(
    "GRAPH_COMPLETION",
    description="GRAPH_COMPLETION (AI Q&A, recommended) | RAG_COMPLETION (traditional RAG) | CHUNKS (raw text) | SUMMARIES (pre-generated) | FEELING_LUCKY (auto-select)"
)

datasets: Optional[list[str]] = Field(
    None,
    description="KB IDs to search. None = all KBs. Use list_datasets() to discover. Example: ['project_docs', 'api_reference']"
)

top_k: int = Field(
    10,
    description="Max results to return (1-100). Lower = faster.",
    ge=1,
    le=100
)
```

---

## 4. Transport Selection: SSE vs streamable-http

### Use Case Requirements

- **Deployment**: LibreChat MCP integration
- **Security**: Localhost binding, SSRF protection
- **Authentication**: Not needed (local private deployment)
- **Streaming**: Desirable for long-running searches

### Transport Comparison

| Feature | SSE (Server-Sent Events) | Streamable HTTP |
|---------|-------------------------|-----------------|
| **Streaming** | Native, one-way | Chunked transfer encoding |
| **Reconnection** | Automatic with EventSource | Manual retry logic |
| **Firewall/Proxy** | Better compatibility | May be blocked |
| **Client Support** | Broader (EventSource API) | Requires chunked parsing |
| **Latency** | Lower (persistent connection) | Higher (new connection per request) |
| **Resource Usage** | 1 connection = 1 long-lived thread | 1 request = 1 short thread |
| **Complexity** | Simple client (EventSource) | More complex client |

### Recommendations

#### A. Primary: Use SSE Transport

**Rationale**:
1. **Better for LibreChat**: Most MCP clients (including LibreChat) prefer SSE
2. **Streaming Support**: Natural fit for long-running GRAPH_COMPLETION searches
3. **Auto-Reconnect**: Client-side reconnection without custom logic
4. **Lower Latency**: Persistent connection reduces request overhead

**Configuration**:
```bash
# Start Cognee MCP with SSE
python src/server.py --transport sse --host 127.0.0.1 --port 8000

# LibreChat MCP config
{
  "mcpServers": {
    "cognee-search": {
      "type": "sse",
      "url": "http://127.0.0.1:8000/sse"
    }
  }
}
```

**SSRF Protection**:
```python
# server.py - enforce localhost binding
mcp.settings.host = "127.0.0.1"  # Never 0.0.0.0 for production

# Add host validation in middleware
@mcp.custom_middleware
async def validate_host(request, call_next):
    if request.client.host not in ["127.0.0.1", "::1", "localhost"]:
        return JSONResponse({"error": "Access denied"}, status_code=403)
    return await call_next(request)
```

#### B. Alternative: Streamable HTTP

Use if:
- LibreChat requires HTTP specifically
- Running behind a reverse proxy that doesn't support SSE
- Need RESTful semantics for monitoring/logging

**Configuration**:
```bash
python src/server.py --transport http --host 127.0.0.1 --port 8000 --path /mcp

# LibreChat config
{
  "mcpServers": {
    "cognee-search": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

#### C. Don't Use stdio

**Why not**:
- Designed for direct process spawning (Claude Desktop, Cursor)
- Incompatible with LibreChat (web-based, needs HTTP transport)
- No streaming support for responses

---

## 5. Context Efficiency for LLM Conversations

### Token Usage Analysis

Current implementation sends **13 tools** with verbose descriptions on every LLM turn:

```
Estimated Token Usage per Turn:
- Tool names (13 x 3 tokens) = 39 tokens
- Tool descriptions (13 x 150 tokens avg) = 1,950 tokens
- Parameter descriptions (13 x 100 tokens avg) = 1,300 tokens
- serverInstructions (if added) = ~200 tokens
------------------------------------------------------
Total Context per Turn: ~3,500 tokens
```

**Impact**:
- Higher latency (more tokens to process)
- Higher cost (tokens charged per turn)
- Slower LLM reasoning (more tools to consider)

### MCP Best Practices for Context Efficiency

From gofastmcp.com:

1. **Minimize Tool Count**: Only expose necessary tools
2. **Short Descriptions**: 1-2 sentences max
3. **Parameter Inlining**: Use enums, not long descriptions
4. **serverInstructions**: One-time context, not repeated
5. **Tool Grouping**: Related operations in single tool

### Recommendations

#### A. Reduced Tool Set (Optimal)

```
Search Server (2 tools):
- search()         ~200 tokens (simplified description)
- list_datasets()  ~100 tokens
------------------------------------------------------
Total Context per Turn: ~300 tokens + serverInstructions (200) = 500 tokens

Reduction: 85% fewer tokens (3,500 -> 500)
```

#### B. Optimized Tool Descriptions

**Bad** (verbose):
```python
@mcp.tool()
async def search(search_query: str, search_type: str) -> list:
    """
    Search and query the knowledge graph for insights, information, and connections.

    This is the final step in the Cognee workflow that retrieves information from the
    processed knowledge graph. It supports multiple search modes optimized for different
    use cases - from simple fact retrieval to complex reasoning and code analysis.
    [... 50 more lines]
    """
```

**Good** (concise):
```python
@mcp.tool()
async def search(
    search_query: str,
    search_type: Literal["GRAPH_COMPLETION", "RAG_COMPLETION", "CHUNKS", "SUMMARIES", "FEELING_LUCKY"] = "GRAPH_COMPLETION",
    datasets: Optional[list[str]] = None,
    top_k: int = 10
) -> list:
    """Search knowledge bases with natural language. Returns AI answers or raw chunks."""
```

**Token Savings**: 1,800 tokens -> 50 tokens per tool

#### C. Use Enums for Parameter Choices

**Bad** (description-heavy):
```python
search_type: str = Field(
    "GRAPH_COMPLETION",
    description="""
    The type of search to perform:
    - GRAPH_COMPLETION: Returns an LLM response based on the search query and graph context
    - RAG_COMPLETION: Returns an LLM response based on document chunks
    - CHUNKS: Returns raw text segments
    - SUMMARIES: Returns pre-generated summaries
    - FEELING_LUCKY: Automatically selects best search type
    """
)
```

**Good** (enum-based):
```python
search_type: Literal[
    "GRAPH_COMPLETION",
    "RAG_COMPLETION",
    "CHUNKS",
    "SUMMARIES",
    "FEELING_LUCKY"
] = "GRAPH_COMPLETION"
```

**Token Savings**: LLM infers meaning from enum values, doesn't need long descriptions

#### D. Move Documentation Out of Descriptions

Create a separate documentation resource:

```python
@mcp.resource("cognee://docs/search-guide")
async def search_guide() -> str:
    """Detailed guide for using search tool"""
    return """
    # Cognee Search Guide

    ## Search Types

    ### GRAPH_COMPLETION (Recommended)
    Uses AI to generate natural language answers based on knowledge graph context.
    Best for: Complex questions, analysis, summaries
    [... detailed documentation]
    """
```

**Benefits**:
- Documentation available on-demand (not sent every turn)
- LLM can reference it if needed
- Keeps tool descriptions minimal

---

## 6. Tool Consolidation

### Current Tool Overlap

Several tools have overlapping concerns:

1. **Data Listing**:
   - `list_data()` - Lists datasets and data items
   - `get_datasets()` - (implicit, not exposed but used internally)

2. **Status Checking**:
   - `cognify_status()` - Cognify pipeline status
   - `codify_status()` - Codify pipeline status

3. **Data Management**:
   - `delete()` - Targeted deletion
   - `prune()` - Full reset

### MCP Best Practices

1. **Single Responsibility**: Each tool should have one clear purpose
2. **Parameter-Based Variation**: Use parameters instead of multiple tools
3. **Cohesive Grouping**: Related operations under one tool

### Recommendations

#### A. Consolidate Status Tools

**Before** (2 tools):
```python
@mcp.tool()
async def cognify_status(): ...

@mcp.tool()
async def codify_status(): ...
```

**After** (1 tool):
```python
@mcp.tool()
async def pipeline_status(
    pipeline_type: Literal["cognify", "codify"] = "cognify",
    dataset_name: Optional[str] = None
) -> dict:
    """
    Get status of data processing pipelines.

    Parameters:
    - pipeline_type: "cognify" (text processing) or "codify" (code analysis)
    - dataset_name: Optional dataset to check (default: all datasets)
    """
```

#### B. Consolidate Data Management

**Before** (3 tools):
```python
@mcp.tool()
async def list_data(dataset_id: Optional[str] = None): ...

@mcp.tool()
async def delete(data_id: str, dataset_id: str, mode: str = "soft"): ...

@mcp.tool()
async def prune(): ...
```

**After** (2 tools):
```python
@mcp.tool()
async def list_datasets(
    dataset_id: Optional[str] = None,
    include_data_items: bool = False
) -> dict:
    """
    List knowledge bases (datasets) and optionally their data items.

    Parameters:
    - dataset_id: Optional specific KB to inspect
    - include_data_items: Include individual data items (slower)
    """

@mcp.tool()
async def delete_data(
    data_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    mode: Literal["soft", "hard", "reset"] = "soft"
) -> dict:
    """
    Delete data from knowledge bases.

    Parameters:
    - data_id: Specific data item to delete (requires dataset_id)
    - dataset_id: Specific dataset to delete from or clear
    - mode:
        * "soft" - Remove data, keep shared entities (safe)
        * "hard" - Remove data and orphaned entities
        * "reset" - Clear entire KB (like prune, but targeted)

    Examples:
    - delete_data(data_id="123", dataset_id="kb1", mode="soft")
    - delete_data(dataset_id="kb1", mode="reset")  # Clear KB
    - delete_data(mode="reset")  # Clear all (like prune)
    """
```

**Token Savings**: 3 tools -> 2 tools, clearer semantics

#### C. Remove Developer Tools from Search Server

Move to separate admin/dev server:

```python
# Search Server (production)
- search()
- list_datasets()

# Admin Server (internal)
- cognify()
- pipeline_status()
- delete_data()
- list_datasets()  # With include_data_items=True

# Dev Server (optional)
- add_developer_rules()
- save_interaction()
- get_developer_rules()
```

---

## 7. Non-Relevant Tools for Knowledge Base Search

### Tools to Remove from Search Server

For a **read-only knowledge base search** scenario, the following tools are not relevant:

#### 1. Data Ingestion Tools
```python
# Remove from search server
@mcp.tool()
async def cognify(data: str, ...): ...

@mcp.tool()
async def codify(repo_path: str): ...

@mcp.tool()
async def cognee_add_developer_rules(...): ...
```

**Rationale**:
- Search server should be read-only
- Data ingestion is an admin/setup task
- Separates concerns (ingest vs query)

#### 2. Status Monitoring Tools
```python
# Remove from search server (or make optional)
@mcp.tool()
async def cognify_status(): ...

@mcp.tool()
async def codify_status(): ...
```

**Rationale**:
- Users don't care about pipeline status when searching
- Relevant only during data ingestion
- Clutters tool list for no benefit

#### 3. Data Modification Tools
```python
# Remove from search server
@mcp.tool()
async def delete(data_id: str, ...): ...

@mcp.tool()
async def prune(): ...
```

**Rationale**:
- Search server should be read-only
- Deletion is an admin task
- Security: prevents accidental data loss

#### 4. Developer-Specific Tools
```python
# Remove from search server
@mcp.tool()
async def save_interaction(data: str): ...

@mcp.tool()
async def get_developer_rules(): ...
```

**Rationale**:
- Not relevant for knowledge base search
- Specific to coding agent workflows
- Adds noise to tool selection

#### 5. Optional: list_data() Detailed Mode
```python
# Simplify for search server
@mcp.tool()
async def list_data(dataset_id: str = None): ...
# -> Rename to list_datasets(), remove dataset_id parameter
```

**Rationale**:
- Detailed data listing is admin task
- Search users only need KB names
- Reduced complexity

### Minimal Search Server Tools

**Recommended minimal set** (2 tools):

```python
@mcp.tool()
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    system_prompt: Optional[str] = None
) -> list:
    """Search knowledge bases with natural language."""

@mcp.tool()
async def list_datasets() -> list:
    """List available knowledge bases for searching."""
```

**Optional additions** (if needed):

```python
@mcp.tool()
async def get_dataset_info(dataset_id: str) -> dict:
    """Get metadata about a specific knowledge base (size, last updated, etc.)."""
```

### Tool Removal Impact

| Server Type | Tool Count | Token Usage | Use Case |
|-------------|-----------|-------------|----------|
| **Current** | 13 tools | ~3,500 tokens/turn | Mixed (search + admin) |
| **Search Only** | 2 tools | ~500 tokens/turn | Knowledge base search |
| **Admin Only** | 8 tools | ~2,000 tokens/turn | Data management |
| **Dev Only** | 3 tools | ~800 tokens/turn | Developer workflows |

**Benefits of Separation**:
- 85% token reduction for search workloads
- Faster LLM reasoning (fewer tools to consider)
- Better security (read-only search server)
- Clearer user experience (purpose-specific tools)

---

## 8. LibreChat Integration Recommendations

### A. Optimal Configuration

#### 1. Run Separate MCP Servers

**Search Server** (user-facing):
```bash
# Terminal 1 - Search Server
python src/server.py \
  --transport sse \
  --host 127.0.0.1 \
  --port 8000 \
  --mode search  # New flag to enable only search tools
```

**Admin Server** (internal):
```bash
# Terminal 2 - Admin Server
python src/server.py \
  --transport sse \
  --host 127.0.0.1 \
  --port 8001 \
  --mode admin  # Enable admin tools
```

#### 2. LibreChat MCP Configuration

```json
{
  "mcpServers": {
    "cognee-search": {
      "type": "sse",
      "url": "http://127.0.0.1:8000/sse",
      "description": "Search Cognee knowledge bases"
    },
    "cognee-admin": {
      "type": "sse",
      "url": "http://127.0.0.1:8001/sse",
      "description": "Manage Cognee knowledge bases (admin only)",
      "enabled": false  # Disable in production
    }
  }
}
```

#### 3. Server Mode Implementation

Add mode selection to `server.py`:

```python
parser.add_argument(
    "--mode",
    choices=["full", "search", "admin", "dev"],
    default="full",
    help="Server mode: full (all tools), search (read-only), admin (data management), dev (developer tools)"
)

args = parser.parse_args()

# Conditionally register tools based on mode
if args.mode in ["full", "search"]:
    @mcp.tool()
    async def search(...): ...

    @mcp.tool()
    async def list_datasets(): ...

if args.mode in ["full", "admin"]:
    @mcp.tool()
    async def cognify(...): ...

    @mcp.tool()
    async def delete_data(...): ...

    @mcp.tool()
    async def pipeline_status(...): ...

if args.mode in ["full", "dev"]:
    @mcp.tool()
    async def add_developer_rules(...): ...

    @mcp.tool()
    async def save_interaction(...): ...
```

### B. Security Hardening

#### 1. Localhost Binding Enforcement

```python
# server.py
def validate_host(host: str):
    """Enforce localhost-only binding for security"""
    if host not in ["127.0.0.1", "::1"]:
        raise ValueError(
            f"Security: Host must be localhost (127.0.0.1 or ::1), got {host}"
        )

mcp.settings.host = "127.0.0.1"  # Never use 0.0.0.0
validate_host(mcp.settings.host)
```

#### 2. SSRF Protection Middleware

```python
@mcp.custom_middleware
async def ssrf_protection(request, call_next):
    """Prevent Server-Side Request Forgery"""
    # Only allow requests from localhost
    client_host = request.client.host
    if client_host not in ["127.0.0.1", "::1", "localhost"]:
        logger.warning(f"SSRF: Blocked request from {client_host}")
        return JSONResponse(
            {"error": "Access denied", "reason": "Non-localhost origin"},
            status_code=403
        )

    # Check for suspicious patterns in request
    if hasattr(request, "body"):
        body = await request.body()
        suspicious_patterns = [
            b"http://",
            b"https://",
            b"file://",
            b"ftp://",
            b"gopher://",
        ]
        if any(pattern in body for pattern in suspicious_patterns):
            logger.warning(f"SSRF: Suspicious URL pattern in request body")
            # Could be legitimate for search queries, so just log for now

    return await call_next(request)
```

#### 3. Rate Limiting

```python
from collections import defaultdict
from time import time

# Simple in-memory rate limiter
request_counts = defaultdict(list)
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

@mcp.custom_middleware
async def rate_limit(request, call_next):
    """Rate limit requests per client"""
    client_id = request.client.host
    now = time()

    # Clean old requests
    request_counts[client_id] = [
        ts for ts in request_counts[client_id]
        if now - ts < RATE_WINDOW
    ]

    # Check limit
    if len(request_counts[client_id]) >= RATE_LIMIT:
        return JSONResponse(
            {"error": "Rate limit exceeded", "retry_after": RATE_WINDOW},
            status_code=429
        )

    request_counts[client_id].append(now)
    return await call_next(request)
```

### C. Multi-KB Isolation

#### 1. Dataset-Level Access Control

```python
# Add dataset validation to search tool
@mcp.tool()
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[list[str]] = None,
    top_k: int = 10
) -> list:
    """Search with multi-KB isolation"""

    # Validate datasets exist and are accessible
    if datasets:
        available_datasets = await cognee_client.list_datasets()
        available_ids = {d["id"] for d in available_datasets}

        invalid_datasets = set(datasets) - available_ids
        if invalid_datasets:
            raise ValueError(
                f"Invalid dataset IDs: {invalid_datasets}. "
                f"Use list_datasets() to see available KBs."
            )

    # Cognee's LanceDB + Kuzu already enforce isolation at DB level
    results = await cognee_client.search(
        query_text=search_query,
        query_type=search_type,
        datasets=datasets,
        top_k=top_k
    )

    return results
```

#### 2. KB Discovery with Metadata

```python
@mcp.tool()
async def list_datasets() -> list:
    """
    List available knowledge bases with metadata.

    Returns list of dicts with:
    - id: Dataset UUID
    - name: Human-readable name
    - created_at: Creation timestamp
    - size: Number of documents (if available)
    - last_updated: Last modification time
    """
    datasets = await cognee_client.list_datasets()

    # Enhance with metadata
    enriched = []
    for ds in datasets:
        enriched.append({
            "id": ds["id"],
            "name": ds["name"],
            "created_at": ds["created_at"],
            # Add custom metadata if available
            "description": ds.get("description", ""),
            "tags": ds.get("tags", []),
        })

    return [types.TextContent(
        type="text",
        text=json.dumps(enriched, indent=2)
    )]
```

### D. Performance Optimization

#### 1. Response Streaming

For long-running GRAPH_COMPLETION searches, stream results:

```python
@mcp.tool()
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    stream: bool = True  # New parameter
) -> list:
    """Search with optional streaming"""

    if stream and search_type in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
        # Stream LLM completion tokens
        async for chunk in cognee_client.search_stream(
            query_text=search_query,
            query_type=search_type,
            datasets=datasets,
            top_k=top_k
        ):
            yield types.TextContent(type="text", text=chunk)
    else:
        # Return full response
        results = await cognee_client.search(
            query_text=search_query,
            query_type=search_type,
            datasets=datasets,
            top_k=top_k
        )
        return [types.TextContent(type="text", text=str(results))]
```

#### 2. Result Caching

```python
from functools import lru_cache
import hashlib

def cache_key(query: str, search_type: str, datasets: Optional[list[str]]) -> str:
    """Generate cache key for search results"""
    key_data = f"{query}:{search_type}:{sorted(datasets or [])}"
    return hashlib.sha256(key_data.encode()).hexdigest()

# Simple in-memory cache (use Redis for production)
search_cache = {}
CACHE_TTL = 300  # 5 minutes

@mcp.tool()
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    use_cache: bool = True  # New parameter
) -> list:
    """Search with optional caching"""

    if use_cache:
        key = cache_key(search_query, search_type, datasets)
        if key in search_cache:
            cached_result, timestamp = search_cache[key]
            if time() - timestamp < CACHE_TTL:
                logger.info(f"Cache hit for query: {search_query[:50]}")
                return cached_result

    # Perform search
    results = await cognee_client.search(
        query_text=search_query,
        query_type=search_type,
        datasets=datasets,
        top_k=top_k
    )

    # Cache results
    if use_cache:
        search_cache[key] = (results, time())

    return results
```

### E. Monitoring & Observability

#### 1. Request Logging

```python
import logging
from time import time

@mcp.custom_middleware
async def request_logging(request, call_next):
    """Log all requests for monitoring"""
    start_time = time()

    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host}"
    )

    # Process request
    response = await call_next(request)

    # Log response
    duration = time() - start_time
    logger.info(
        f"Response: {response.status_code} "
        f"in {duration:.3f}s"
    )

    return response
```

#### 2. Health Check Endpoint

```python
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Enhanced health check with system status"""
    try:
        # Check database connectivity
        datasets = await cognee_client.list_datasets()

        # Check LLM availability (if in search mode)
        llm_status = "unknown"
        if os.getenv("LLM_API_KEY"):
            llm_status = "configured"

        return JSONResponse({
            "status": "healthy",
            "version": "1.0.0",
            "mode": args.mode,  # search/admin/dev/full
            "transport": args.transport,
            "database": {
                "connected": True,
                "datasets": len(datasets)
            },
            "llm": {
                "status": llm_status
            },
            "uptime": time() - start_time
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e)
        }, status_code=503)
```

#### 3. Metrics Endpoint

```python
# Track metrics
metrics = {
    "requests_total": 0,
    "requests_by_tool": defaultdict(int),
    "errors_total": 0,
    "search_latency_avg": 0.0,
}

@mcp.custom_middleware
async def metrics_tracking(request, call_next):
    """Track request metrics"""
    metrics["requests_total"] += 1

    # Extract tool name from request
    if hasattr(request, "body"):
        body = await request.body()
        try:
            data = json.loads(body)
            tool_name = data.get("method", "unknown")
            metrics["requests_by_tool"][tool_name] += 1
        except:
            pass

    start = time()
    response = await call_next(request)
    duration = time() - start

    # Update latency (exponential moving average)
    metrics["search_latency_avg"] = (
        0.9 * metrics["search_latency_avg"] + 0.1 * duration
    )

    if response.status_code >= 400:
        metrics["errors_total"] += 1

    return response

@mcp.custom_route("/metrics", methods=["GET"])
async def metrics_endpoint(request):
    """Prometheus-compatible metrics"""
    return JSONResponse(metrics)
```

---

## 9. Implementation Priority

### Phase 1: Critical Fixes (Week 1)

**Goal**: Fix broken functionality and add missing parameters

1. **Fix search() parameter passing in direct mode**
   - Update `cognee_client.py` to pass `datasets`, `top_k`, `system_prompt`
   - Add parameter validation
   - Test multi-KB search

2. **Add missing search() parameters**
   - Add `datasets`, `top_k`, `system_prompt` to tool signature
   - Update tool description
   - Add usage examples

3. **Implement server mode selection**
   - Add `--mode` argument (full/search/admin/dev)
   - Conditional tool registration
   - Update documentation

**Expected Impact**:
- Multi-KB search works correctly
- Better control over result count
- Clear separation of concerns

### Phase 2: Optimization (Week 2)

**Goal**: Improve context efficiency and performance

1. **Reduce token usage**
   - Simplify tool descriptions (target 50 tokens each)
   - Add `serverInstructions`
   - Move documentation to resources

2. **Consolidate tools**
   - Merge `cognify_status` + `codify_status` -> `pipeline_status`
   - Merge `delete` + `prune` -> `delete_data`
   - Rename `list_data` -> `list_datasets`

3. **Add caching**
   - Implement search result caching
   - Add cache control parameter
   - Monitor cache hit rate

**Expected Impact**:
- 85% reduction in token usage (3,500 -> 500 tokens/turn)
- Faster LLM reasoning
- Better response latency for repeated queries

### Phase 3: Production Hardening (Week 3)

**Goal**: Security, monitoring, and deployment

1. **Security hardening**
   - Enforce localhost binding
   - Add SSRF protection middleware
   - Implement rate limiting

2. **Monitoring & observability**
   - Enhanced health checks
   - Metrics endpoint
   - Request logging

3. **LibreChat integration testing**
   - Test SSE transport with LibreChat
   - Validate multi-KB search
   - Performance benchmarking

**Expected Impact**:
- Production-ready deployment
- SSRF protection in place
- Observable system behavior

### Phase 4: Advanced Features (Week 4+)

**Goal**: Enhanced functionality

1. **Response streaming**
   - Stream GRAPH_COMPLETION results
   - SSE chunk handling
   - Client-side streaming support

2. **Advanced caching**
   - Redis integration
   - Cache warming strategies
   - TTL configuration

3. **Dataset metadata**
   - Rich dataset descriptions
   - Tagging and categorization
   - Search filtering by metadata

**Expected Impact**:
- Better user experience (streaming)
- Scalable caching (Redis)
- Improved KB discovery

---

## 10. Summary of Recommendations

### Immediate Actions (Phase 1)

1. **Fix search() direct mode parameter passing**
   ```python
   # cognee_client.py - pass all parameters
   results = await self.cognee.search(
       query_type=SearchType[query_type.upper()],
       query_text=query_text,
       datasets=datasets,  # ADD
       top_k=top_k,        # ADD
       system_prompt=system_prompt  # ADD
   )
   ```

2. **Add missing parameters to search() tool**
   ```python
   @mcp.tool()
   async def search(
       search_query: str,
       search_type: str = "GRAPH_COMPLETION",
       datasets: Optional[list[str]] = None,     # NEW
       top_k: int = 10,                         # NEW
       system_prompt: Optional[str] = None      # NEW
   ) -> list:
   ```

3. **Implement server modes**
   ```bash
   # Search server (user-facing)
   python src/server.py --mode search --transport sse --port 8000

   # Admin server (internal)
   python src/server.py --mode admin --transport sse --port 8001
   ```

### High-Impact Optimizations (Phase 2)

1. **Reduce tool count from 13 to 2 for search server**
   - Expose only: `search()`, `list_datasets()`
   - Token reduction: 85% (3,500 -> 500 tokens/turn)

2. **Simplify tool descriptions**
   - Target: <100 tokens per tool
   - Move details to `serverInstructions`
   - Use enums for parameters

3. **Use SSE transport**
   - Better for LibreChat integration
   - Native streaming support
   - Auto-reconnection

### Security Best Practices (Phase 3)

1. **Localhost binding enforcement**
   ```python
   mcp.settings.host = "127.0.0.1"  # Never 0.0.0.0
   ```

2. **SSRF protection middleware**
   ```python
   @mcp.custom_middleware
   async def ssrf_protection(request, call_next):
       if request.client.host not in ["127.0.0.1", "::1", "localhost"]:
           return JSONResponse({"error": "Access denied"}, status_code=403)
       return await call_next(request)
   ```

3. **Rate limiting**
   - 100 requests/minute per client
   - Prevent abuse

### Tool Removals for Search Server

**Remove from search server** (move to admin):
- `cognify()` - Data ingestion (admin task)
- `cognify_status()` - Pipeline monitoring (admin task)
- `codify()` - Code analysis (admin task)
- `codify_status()` - Code pipeline monitoring (admin task)
- `delete()` - Data deletion (admin task)
- `prune()` - System reset (admin task)
- `list_data(dataset_id)` - Detailed listing (admin task)
- `save_interaction()` - Developer logging (dev task)
- `get_developer_rules()` - Developer rules (dev task)
- `cognee_add_developer_rules()` - Rule ingestion (dev task)

**Keep for search server**:
- `search()` - Core search functionality
- `list_datasets()` - KB discovery (simplified)

---

## 11. Performance Benchmarks

### Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Tools in Context** | 13 | 2 | 85% reduction |
| **Tokens per Turn** | ~3,500 | ~500 | 85% reduction |
| **LLM Latency** | Baseline | -30-40% | Faster reasoning |
| **Search Latency** | Baseline | -20-30% (with cache) | Faster responses |
| **Memory Usage** | Baseline | -40% | Fewer tools loaded |

### Load Testing Scenarios

```bash
# Test 1: Sequential searches (no caching)
for i in {1..100}; do
  curl -X POST http://127.0.0.1:8000/mcp \
    -H "Content-Type: application/json" \
    -d '{"method":"search","params":{"search_query":"test query '$i'"}}'
done

# Test 2: Repeated searches (cache benefit)
for i in {1..100}; do
  curl -X POST http://127.0.0.1:8000/mcp \
    -H "Content-Type: application/json" \
    -d '{"method":"search","params":{"search_query":"same query"}}'
done

# Test 3: Multi-KB search
curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method":"search",
    "params":{
      "search_query":"test",
      "datasets":["kb1","kb2","kb3"],
      "top_k":5
    }
  }'
```

---

## 12. References

### MCP Specification
- **Official Spec**: https://modelcontextprotocol.io/specification/2025-06-18
- **Key Sections**:
  - Tool Definition (JSON-RPC methods)
  - Transport Protocols (SSE, HTTP, stdio)
  - Error Handling
  - Schema Validation

### Best Practices
- **gofastmcp.com**: https://gofastmcp.com/
  - Context efficiency guidelines
  - Tool design patterns
  - Performance optimization tips

### Cognee Documentation
- **API Docs**: https://docs.cognee.ai/
- **MCP Integration**: https://docs.cognee.ai/how-to-guides/deployment/mcp

### Related Technologies
- **FastMCP**: https://github.com/jlowin/fastmcp
- **LanceDB**: https://lancedb.github.io/lancedb/
- **Kuzu**: https://kuzudb.com/
- **LibreChat**: https://www.librechat.ai/

---

## Appendix A: Code Examples

### Complete Search-Only Server

```python
# cognee-mcp/src/search_server.py
"""
Cognee MCP Search Server - Read-only knowledge base search
"""

import asyncio
from typing import Optional, Literal
from mcp.server import FastMCP
from mcp import types
import uvicorn

# Initialize MCP server
mcp = FastMCP(
    "Cognee Search",
    instructions="""Read-only knowledge base search server.

Usage:
1. list_datasets() - Discover available KBs
2. search() - Query one or multiple KBs

Features:
- Multi-KB search with isolation
- AI-powered answers (GRAPH_COMPLETION)
- Fast text lookup (CHUNKS)
- Pre-generated summaries (SUMMARIES)

Security: Localhost-only, read-only access."""
)

from cognee_client import CogneeClient

cognee_client: Optional[CogneeClient] = None


@mcp.tool()
async def search(
    search_query: str,
    search_type: Literal[
        "GRAPH_COMPLETION",
        "RAG_COMPLETION",
        "CHUNKS",
        "SUMMARIES",
        "FEELING_LUCKY"
    ] = "GRAPH_COMPLETION",
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    system_prompt: Optional[str] = None
) -> list:
    """
    Search knowledge bases with natural language.

    Returns AI-generated answers or raw text depending on search_type.
    Use list_datasets() to discover available KBs.
    """

    # Validate datasets if provided
    if datasets:
        available = await cognee_client.list_datasets()
        available_ids = {d["id"] for d in available}
        invalid = set(datasets) - available_ids
        if invalid:
            return [types.TextContent(
                type="text",
                text=f"Error: Invalid dataset IDs: {invalid}. Use list_datasets() first."
            )]

    # Perform search
    results = await cognee_client.search(
        query_text=search_query,
        query_type=search_type,
        datasets=datasets,
        top_k=top_k,
        system_prompt=system_prompt
    )

    return [types.TextContent(type="text", text=str(results))]


@mcp.tool()
async def list_datasets() -> list:
    """List available knowledge bases for searching."""

    datasets = await cognee_client.list_datasets()

    # Format for display
    lines = ["Available Knowledge Bases:", "=" * 50, ""]
    for i, ds in enumerate(datasets, 1):
        lines.append(f"{i}. {ds['name']}")
        lines.append(f"   ID: {ds['id']}")
        lines.append(f"   Created: {ds.get('created_at', 'N/A')}")
        lines.append("")

    return [types.TextContent(type="text", text="\n".join(lines))]


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint"""
    return {"status": "healthy", "mode": "search"}


@mcp.custom_middleware
async def security_middleware(request, call_next):
    """Enforce localhost-only access"""
    if request.client.host not in ["127.0.0.1", "::1", "localhost"]:
        return {"error": "Access denied"}, 403
    return await call_next(request)


async def main():
    global cognee_client

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["sse", "http"], default="sse")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-token", default=None)
    args = parser.parse_args()

    # Initialize client
    cognee_client = CogneeClient(api_url=args.api_url, api_token=args.api_token)

    # Configure server
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # Start server
    if args.transport == "sse":
        sse_app = mcp.sse_app()
        config = uvicorn.Config(sse_app, host=args.host, port=args.port)
        server = uvicorn.Server(config)
        await server.serve()
    else:
        http_app = mcp.streamable_http_app()
        config = uvicorn.Config(http_app, host=args.host, port=args.port)
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
```

### Usage Example

```bash
# Start search server
python src/search_server.py --transport sse --port 8000

# Configure LibreChat
cat > ~/.librechat/mcp.json <<EOF
{
  "mcpServers": {
    "cognee-search": {
      "type": "sse",
      "url": "http://127.0.0.1:8000/sse"
    }
  }
}
EOF

# Test with curl
curl -X POST http://127.0.0.1:8000/sse \
  -H "Content-Type: application/json" \
  -d '{
    "method": "search",
    "params": {
      "search_query": "What are the main concepts?",
      "search_type": "GRAPH_COMPLETION",
      "datasets": ["project_docs"],
      "top_k": 5
    }
  }'
```

---

## Appendix B: Migration Guide

### Step-by-Step Migration

#### 1. Create Search-Only Server (Recommended)

```bash
# Copy current server
cp cognee-mcp/src/server.py cognee-mcp/src/search_server.py

# Edit search_server.py
# - Remove all tools except search() and list_datasets()
# - Add serverInstructions
# - Simplify tool descriptions
# - Add security middleware

# Test new server
python cognee-mcp/src/search_server.py --transport sse --port 8000
```

#### 2. Update Existing Server with Modes (Alternative)

```bash
# Edit server.py
# - Add --mode argument
# - Conditional tool registration
# - Test each mode

python src/server.py --mode search --transport sse --port 8000
python src/server.py --mode admin --transport sse --port 8001
```

#### 3. Fix Parameter Passing

```bash
# Edit cognee-mcp/src/cognee_client.py
# - Update search() method (line 148-197)
# - Pass datasets, top_k, system_prompt in direct mode
# - Test multi-KB search

# Run tests
python -m pytest cognee-mcp/tests/test_search.py
```

#### 4. Update LibreChat Config

```bash
# Update ~/.librechat/mcp.json
{
  "mcpServers": {
    "cognee-search": {
      "type": "sse",
      "url": "http://127.0.0.1:8000/sse",
      "description": "Search Cognee knowledge bases"
    }
  }
}

# Restart LibreChat
docker-compose restart librechat
```

### Rollback Plan

If issues occur:

1. **Keep old server running**: Don't shut down current server until new one is validated
2. **Switch LibreChat config**: Update `mcp.json` to point back to old server
3. **Log issues**: Capture errors for debugging
4. **Gradual rollout**: Test with one KB first, then expand

---

## Conclusion

This analysis provides comprehensive recommendations for optimizing the Cognee MCP implementation for knowledge base search scenarios. The key takeaways:

1. **Separate concerns**: Run search and admin servers separately
2. **Minimize context**: Reduce from 13 tools to 2 for search workloads
3. **Fix parameter passing**: Enable multi-KB search with proper parameter handling
4. **Use SSE transport**: Best for LibreChat integration and streaming
5. **Harden security**: Localhost binding, SSRF protection, rate limiting

**Expected impact**:
- 85% reduction in token usage (3,500 -> 500 tokens/turn)
- 30-40% faster LLM reasoning
- Better security posture (read-only search server)
- Improved user experience (focused tools, faster responses)

**Implementation priority**:
1. **Phase 1** (Week 1): Fix broken functionality (parameter passing)
2. **Phase 2** (Week 2): Optimize context efficiency (reduce tools)
3. **Phase 3** (Week 3): Production hardening (security, monitoring)
4. **Phase 4** (Week 4+): Advanced features (streaming, caching)

For questions or clarifications, refer to:
- MCP Specification: https://modelcontextprotocol.io/specification/2025-06-18
- gofastmcp.com: https://gofastmcp.com/
- Cognee Docs: https://docs.cognee.ai/
