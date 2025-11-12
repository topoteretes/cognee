# Cognee MCP - Implementation Plan
**Keep Only Jina + Phases 1-3**

**Date:** 2025-01-12
**Status:** Ready for Implementation
**Total Estimated Time:** 3-4 days (24-32 hours)

---

## Table of Contents
1. [Keep Only Jina - Remove LocalReranker](#keep-only-jina)
2. [Phase 1: Critical Bugs](#phase-1-critical-bugs)
3. [Phase 2: High-Priority Improvements](#phase-2-high-priority)
4. [Phase 3: Code Quality & Security](#phase-3-code-quality)
5. [Testing Strategy](#testing-strategy)
6. [Rollback Plan](#rollback-plan)

---

## Keep Only Jina - Remove LocalReranker

**Priority:** MEDIUM (solves 2 critical bugs by removal)
**Estimated Time:** 1-2 hours
**Impact:** Simplifies codebase, removes 100+ lines, fixes 2 critical bugs

### Rationale
- User's personal use case never exceeds Jina's free tier (500 RPM)
- Better search quality than local sentence-transformers
- Simpler codebase (no dual reranking code paths)
- Fixes critical bug: LocalReranker missing `__aenter__`/`__aexit__`
- Fixes performance bug: Model reloading on every call
- Jina free tier = $0/month forever for personal use

### All LocalReranker References Found

**Via Serena Analysis:**
1. `cognee-mcp/src/reranker.py:165-230` - LocalReranker class definition
2. `cognee-mcp/src/reranker.py:255-256` - RerankerFactory.create_reranker()
3. `cognee-mcp/src/tools.py:517` - Documentation mentions "local" option
4. `cognee-mcp/src/tools.py:613-653` - Local reranker implementation block

### Step-by-Step Removal

#### Step 1: Remove LocalReranker Class
**File:** `cognee-mcp/src/reranker.py`
**Lines:** 165-230 (66 lines)

**Action:** DELETE entire LocalReranker class

```python
# DELETE LINES 165-230
# class LocalReranker:
#     """
#     Local sentence-transformers reranker fallback.
#     ...
#     """
#     # ... entire implementation ...
```

**Result:** File reduced from 262 → 196 lines

---

#### Step 2: Update RerankerFactory
**File:** `cognee-mcp/src/reranker.py`
**Lines:** 239-262

**Current Code:**
```python
@staticmethod
def create_reranker(provider: str, api_key: Optional[str] = None):
    """
    Create a reranker based on provider type.

    Args:
        provider: 'jina', 'local', or 'off'
        api_key: Optional API key (for jina provider)

    Returns:
        Reranker instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider == "jina":
        return JinaReranker(api_key)
    elif provider == "local":
        return LocalReranker()
    elif provider == "off":
        return None
    else:
        raise ValueError(
            f"Unsupported rerank provider: {provider}. Use 'jina', 'local', or 'off'"
        )
```

**Updated Code:**
```python
@staticmethod
def create_reranker(provider: str, api_key: Optional[str] = None):
    """
    Create a reranker based on provider type.

    Args:
        provider: 'jina' or 'off'
        api_key: API key for Jina AI (required if provider='jina')

    Returns:
        Reranker instance or None

    Raises:
        ValueError: If provider is not supported or API key missing
    """
    if provider == "jina":
        if not api_key:
            raise ValueError(
                "JINA_API_KEY environment variable required for Jina reranking. "
                "Get free API key: https://jina.ai/?sui=apikey"
            )
        return JinaReranker(api_key)
    elif provider == "off":
        return None
    else:
        raise ValueError(
            f"Unsupported rerank provider: {provider}. Use 'jina' or 'off'"
        )
```

**Changes:**
- ✅ Removed "local" option from docstring
- ✅ Removed `elif provider == "local"` branch
- ✅ Added API key validation for Jina
- ✅ Updated error message to only mention "jina" and "off"

---

#### Step 3: Remove Local Reranker Usage Block
**File:** `cognee-mcp/src/tools.py`
**Lines:** 613-653 (41 lines)

**Action:** DELETE entire `elif rerank_provider == "local":` block

**Current Code:**
```python
                            elif rerank_provider == "local":
                                # Local sentence-transformers fallback
                                async with reranker_instance as reranker:
                                    logger.info("Using local sentence-transformers reranker")

                                    rerank_results = await reranker.rerank(
                                        query=search_query,
                                        documents=documents,
                                        top_n=min(top_k, len(documents)),
                                    )

                                    # Map back to original results
                                    if rerank_results:
                                        scored_results = []
                                        for item in rerank_results:
                                            idx = item["index"]
                                            score = item["relevance_score"]

                                            if idx < len(search_results):
                                                result = search_results[idx].copy()
                                                result.update(
                                                    {
                                                        "mcp_relevance_score": float(score),
                                                        "backend_score": result.get("score"),
                                                        "ranking_method": "local_sentence_transformers",
                                                    }
                                                )
                                                scored_results.append(result)

                                        # Sort by similarity score
                                        scored_results.sort(
                                            key=lambda x: x["mcp_relevance_score"], reverse=True
                                        )
                                        search_results = scored_results[:top_k]

                                        logger.info(
                                            f"Locally reranked {len(search_results)} results"
                                        )
                                    else:
                                        logger.warning("Local reranker returned no results")
                                        search_results = search_results[:top_k]
```

**After Deletion:** Lines 613-653 removed entirely. Flow goes directly from Jina block (line 612) to else block (line 654).

---

#### Step 4: Update search() Tool Documentation
**File:** `cognee-mcp/src/tools.py`
**Line:** 517

**Current:**
```python
    - rerank_provider: jina (AI reranker) | local (sentence-transformers) | off (no reranking)
```

**Updated:**
```python
    - rerank_provider: jina (AI reranker, requires JINA_API_KEY) | off (no reranking)
```

**Also Update:** Lines 499 (parameter signature) if using Literal type hints (covered in Phase 2)

---

### Verification Steps

After removal, verify:

1. **Reranker module compiles:**
   ```bash
   python3 -m py_compile cognee-mcp/src/reranker.py
   ```

2. **Tools module compiles:**
   ```bash
   python3 -m py_compile cognee-mcp/src/tools.py
   ```

3. **Search with Jina works:**
   ```bash
   # Start server
   python cognee-mcp/src/server.py --transport sse

   # In another terminal, test search
   # (LibreChat or MCP client)
   search(search_query="test", rerank_provider="jina")
   ```

4. **Search with rerank=off works:**
   ```bash
   search(search_query="test", rerank_provider="off")
   ```

5. **Attempting "local" raises clear error:**
   ```python
   # Should raise: ValueError: Unsupported rerank provider: local. Use 'jina' or 'off'
   search(search_query="test", rerank_provider="local")
   ```

---

### Files Modified Summary

| File | Lines Changed | Description |
|------|---------------|-------------|
| `reranker.py` | -66 lines (165-230) | Remove LocalReranker class |
| `reranker.py` | ~15 lines (239-262) | Update RerankerFactory |
| `tools.py` | -41 lines (613-653) | Remove local reranker block |
| `tools.py` | 1 line (517) | Update docstring |

**Total Lines Removed:** 107 lines
**Total Complexity Reduction:** 1 class, 1 conditional branch, 2 critical bugs

---

## Phase 1: Critical Bugs

**Priority:** CRITICAL
**Estimated Time:** 4-6 hours
**Must Complete Before:** Production deployment

### Overview

5 critical bugs that will cause runtime crashes or protocol failures:
1. ~~LocalReranker missing context manager~~ (SOLVED by removal above)
2. Stdio transport stdout corruption
3. HTTP client resource leak
4. Tool return type annotations wrong
5. API parameter format mismatch

### Bug 1: ~~LocalReranker Missing Context Manager~~ ✅ SOLVED

**Status:** SOLVED by "Keep Only Jina" removal
**No action needed** - This bug is fixed by removing LocalReranker entirely.

---

### Bug 2: Stdio Transport Stdout Corruption

**Severity:** CRITICAL
**Impact:** Corrupts MCP protocol stream, causes connection failures
**Estimated Time:** 30 minutes

#### Problem
Server initialization logs to stdout without redirection when using stdio transport, violating MCP specification.

**Location:** `cognee-mcp/src/server.py:78-188`

#### Current Code
```python
async def main():
    """Main entry point for the Cognee MCP server."""
    parser = argparse.ArgumentParser(description="Cognee MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["stdio", "sse", "http"],
        help="Transport type for the MCP server",
    )
    # ... more arguments ...

    args = parser.parse_args()

    # MISSING: Stdout redirection for stdio transport

    # These functions log to stdout - CORRUPTS PROTOCOL!
    validate_settings()
    await health.perform_startup_health_check()
    # ...
```

#### Fix
```python
async def main():
    """Main entry point for the Cognee MCP server."""
    parser = argparse.ArgumentParser(description="Cognee MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["stdio", "sse", "http"],
        help="Transport type for the MCP server",
    )
    # ... more arguments ...

    args = parser.parse_args()

    # ✅ CRITICAL FIX: Redirect stdout to stderr for stdio transport
    if args.transport == "stdio":
        import sys
        sys.stdout = sys.stderr
        logger.info("Stdout redirected to stderr for stdio transport (MCP protocol compliance)")

    # Now safe to proceed with initialization
    validate_settings()
    await health.perform_startup_health_check()
    # ...
```

#### Testing
```bash
# Test stdio transport doesn't corrupt protocol
python cognee-mcp/src/server.py --transport stdio

# Should see logs on stderr, not stdout
# MCP protocol stream on stdout should be clean
```

**File Modified:** `cognee-mcp/src/server.py` (add 4 lines after argument parsing)

---

### Bug 3: HTTP Client Resource Leak

**Severity:** CRITICAL
**Impact:** Memory leaks, connection pool exhaustion
**Estimated Time:** 15 minutes

#### Problem
If `aclose()` is interrupted by exception, `self._client` remains non-None, causing resource leak.

**Location:** `cognee-mcp/src/cognee_client.py:89-98`

#### Current Code
```python
async def close(self):
    """Close the HTTP client if in API mode."""
    if self._client:
        await self._client.aclose()  # ❌ If interrupted, _client stays non-None
        logger.debug("HTTP client closed")
```

#### Fix
```python
async def close(self):
    """Close the HTTP client if in API mode."""
    if self._client:
        try:
            await self._client.aclose()
            logger.debug("HTTP client closed")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")
        finally:
            self._client = None  # ✅ Always reset, even on error
```

#### Testing
```python
# Simulate error during close
async def test_close_with_error():
    client = CogneeClient(use_api=True)

    # Mock aclose to raise error
    original_aclose = client._client.aclose
    async def error_aclose():
        raise RuntimeError("Network error during close")
    client._client.aclose = error_aclose

    await client.close()

    # Verify _client is None despite error
    assert client._client is None, "Client should be None after close"
```

**File Modified:** `cognee-mcp/src/cognee_client.py:89-98` (add try/finally block)

---

### Bug 4: Tool Return Type Annotations Wrong

**Severity:** CRITICAL (protocol compliance)
**Impact:** Misleading type hints, poor schema generation
**Estimated Time:** 2 hours

#### Problem
All 12 tools return `list` instead of `List[types.TextContent]`, violating MCP type contracts.

**Location:** `cognee-mcp/src/tools.py` (multiple functions)

#### All Functions Requiring Fix

| Line | Function | Current Return Type | Fixed Return Type |
|------|----------|---------------------|-------------------|
| 72 | `cognify()` | `-> list:` | `-> List[types.TextContent]:` |
| 170 | `cognify_status()` | `-> list:` | `-> List[types.TextContent]:` |
| 341 | `add()` | `-> list:` | `-> List[types.TextContent]:` |
| 412 | `delete()` | `-> list:` | `-> List[types.TextContent]:` |
| 491 | `search()` | `-> list:` | `-> List[types.TextContent]:` |
| 712 | `list_datasets()` | `-> list:` | `-> List[types.TextContent]:` |
| 792 | `prune()` | `-> list:` | `-> List[types.TextContent]:` |
| 846 | `save_interaction()` | `-> list:` | `-> List[types.TextContent]:` |
| 906 | `get_developer_rules()` | `-> list:` | `-> List[types.TextContent]:` |

#### Implementation Steps

**1. Add import at top of file** (after line 1):
```python
from typing import List, Optional
import mcp.types as types
```

**2. Update each function signature:**

**Example for `cognify()`** (line 72):
```python
# BEFORE
async def cognify(...) -> list:

# AFTER
async def cognify(...) -> List[types.TextContent]:
```

**Example for `search()`** (line 491):
```python
# BEFORE
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[List[str]] = None,
    top_k: int = 10,
    system_prompt: Optional[str] = None,
    rerank: bool = True,
    rerank_provider: str = "jina",
    rerank_model: str = "jina-reranker-v2-base-multilingual",
) -> list:

# AFTER
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[List[str]] = None,
    top_k: int = 10,
    system_prompt: Optional[str] = None,
    rerank: bool = True,
    rerank_provider: str = "jina",  # Will be Literal in Phase 2
    rerank_model: str = "jina-reranker-v2-base-multilingual",
) -> List[types.TextContent]:
```

#### Automated Fix Script

```python
# fix_return_types.py
import re

file_path = "cognee-mcp/src/tools.py"

# Read file
with open(file_path, 'r') as f:
    content = f.read()

# Add imports if not present
if "from typing import List" not in content:
    content = "from typing import List, Optional\nimport mcp.types as types\n\n" + content

# Replace all `-> list:` with `-> List[types.TextContent]:`
content = re.sub(r'(\) -> )list:', r'\1List[types.TextContent]:', content)

# Write back
with open(file_path, 'w') as f:
    f.write(content)

print("✅ Fixed all return type annotations")
```

**Run:**
```bash
python fix_return_types.py
python3 -m py_compile cognee-mcp/src/tools.py  # Verify syntax
```

#### Testing
```bash
# Run type checker
python3 -m mypy cognee-mcp/src/tools.py --strict

# Should pass without type errors
```

**File Modified:** `cognee-mcp/src/tools.py` (9 function signatures + imports)

---

### Bug 5: API Parameter Format Mismatch

**Severity:** CRITICAL (for API mode)
**Impact:** Backend may not parse `node_set` correctly
**Estimated Time:** 15 minutes

#### Problem
MCP sends `node_set` as JSON string, but backend expects `List[str]`.

**Location:** `cognee-mcp/src/cognee_client.py:136`

#### Current Code
```python
# Line 133-137
if node_set:
    for item in node_set:
        form_data.append(("node_set", item))
else:
    form_data["node_set"] = json.dumps(node_set)  # ❌ WRONG - sends JSON string
```

#### Fix
```python
# Line 133-137
if node_set:
    for item in node_set:
        form_data.append(("node_set", item))
# ✅ REMOVE the else block entirely - let FastAPI handle None
```

**Rationale:** FastAPI's multipart/form-data handling correctly processes:
- `List[str]`: Appended individually (correct, already done in `if` block)
- `None`: Omitted from form data (FastAPI default behavior)

Sending `json.dumps(None)` = `"null"` string is incorrect.

#### Testing
```python
# Test with API mode
client = CogneeClient(use_api=True, api_url="http://localhost:8000")

# Test with node_set=None
await client.cognify(datasets=["test"], node_set=None)

# Test with node_set=["node1", "node2"]
await client.cognify(datasets=["test"], node_set=["node1", "node2"])

# Verify backend receives correct format
```

**File Modified:** `cognee-mcp/src/cognee_client.py:133-137` (remove 2 lines)

---

### Phase 1 Summary

| Bug | File | Lines Changed | Time | Status |
|-----|------|---------------|------|--------|
| 1. LocalReranker context manager | N/A | N/A | N/A | ✅ Solved by removal |
| 2. Stdio stdout redirect | server.py | +4 | 30 min | ⏳ TODO |
| 3. HTTP client leak | cognee_client.py | +5 | 15 min | ⏳ TODO |
| 4. Return type annotations | tools.py | ~12 | 2 hours | ⏳ TODO |
| 5. API parameter format | cognee_client.py | -2 | 15 min | ⏳ TODO |

**Total Time:** ~3 hours (down from 4-6 hours due to LocalReranker removal)

---

## Phase 2: High-Priority Improvements

**Priority:** HIGH
**Estimated Time:** 8-10 hours
**Should Complete Before:** Production deployment

### Overview

5 high-priority improvements for protocol compliance, performance, and feature completeness:
1. Add Optional/Literal type hints
2. ~~Fix TaskGroup background execution~~ (Document as synchronous)
3. ~~Cache sentence-transformers model~~ (N/A - LocalReranker removed)
4. Fix CORS middleware
5. Add datasetId support

---

### Improvement 1: Add Optional/Literal Type Hints

**Severity:** HIGH
**Impact:** Better schema validation, client autocomplete
**Estimated Time:** 2 hours

#### Problem
Parameters use `str = "default"` instead of `Literal[...]` for enums, and missing `Optional` hints.

**Location:** `cognee-mcp/src/tools.py` (all tool signatures)

#### Examples to Fix

**search() function** (lines 491-501):
```python
# BEFORE
async def search(
    search_query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[List[str]] = None,
    top_k: int = 10,
    system_prompt: Optional[str] = None,
    rerank: bool = True,
    rerank_provider: str = "jina",
    rerank_model: str = "jina-reranker-v2-base-multilingual",
) -> List[types.TextContent]:

# AFTER
async def search(
    search_query: str,
    search_type: Literal[
        "GRAPH_COMPLETION",
        "RAG_COMPLETION",
        "CHUNKS",
        "SUMMARIES",
        "FEELING_LUCKY"
    ] = "GRAPH_COMPLETION",
    datasets: Optional[List[str]] = None,
    top_k: int = 10,
    system_prompt: Optional[str] = None,
    rerank: bool = True,
    rerank_provider: Literal["jina", "off"] = "jina",  # ✅ Removed "local"
    rerank_model: str = "jina-reranker-v2-base-multilingual",
) -> List[types.TextContent]:
```

**cognify() function** (lines 72-81):
```python
# BEFORE
async def cognify(
    datasets: List[str],
    node_set: Optional[List[str]] = None,
) -> List[types.TextContent]:

# AFTER
async def cognify(
    datasets: List[str],
    node_set: Optional[List[str]] = None,  # ✅ Already has Optional
) -> List[types.TextContent]:
```

**delete() function** (lines 412-421):
```python
# BEFORE
async def delete(
    data_id: str = None,
    dataset_name: str = None,
) -> List[types.TextContent]:

# AFTER
async def delete(
    data_id: Optional[str] = None,  # ✅ Add Optional
    dataset_name: Optional[str] = None,  # ✅ Add Optional
) -> List[types.TextContent]:
```

#### All Functions Requiring Type Hint Updates

| Function | Parameters Needing Literal | Parameters Needing Optional |
|----------|---------------------------|----------------------------|
| `cognify()` | None | ✅ Already correct |
| `cognify_status()` | None | dataset_name |
| `add()` | None | dataset_name |
| `delete()` | None | data_id, dataset_name |
| `search()` | search_type, rerank_provider | datasets, system_prompt |
| `list_datasets()` | None | None |
| `prune()` | None | None |
| `save_interaction()` | None | None |
| `get_developer_rules()` | None | None |

#### Implementation

**1. Add import:**
```python
from typing import List, Optional, Literal
```

**2. Update each function signature using table above**

**3. Verify with type checker:**
```bash
python3 -m mypy cognee-mcp/src/tools.py --strict
```

**File Modified:** `cognee-mcp/src/tools.py` (9 function signatures)

---

### Improvement 2: Document TaskGroup as Synchronous

**Severity:** HIGH
**Impact:** Clear user expectations, prevent timeout surprises
**Estimated Time:** 1 hour

#### Problem
Documentation claims "background process" but `async with TaskGroup()` blocks until completion.

**Location:** `cognee-mcp/src/tools.py:311-335, 388-408, 470-488`

#### Decision: Document Synchronous Behavior (Not Implement True Background)

**Rationale:**
- True background tasks require persistent task tracking
- MCP is stateless (no task ID → status mapping)
- Blocking behavior is actually safer (user knows when it's done)
- Most operations complete quickly (<30 seconds)

#### Current Misleading Documentation

**cognify()** (line 68-70):
```python
"""
Process and integrate data into the knowledge graph.
This starts a background process...  # ❌ MISLEADING
"""
```

**add()** (line 337-339):
```python
"""
Add text data or file content to a knowledge base.
Data is queued for processing (use cognify to process).  # ✅ Correct
"""
```

#### Fix: Update Docstrings

**cognify()** (lines 68-81):
```python
@mcp.tool()
async def cognify(
    datasets: List[str],
    node_set: Optional[List[str]] = None,
) -> List[types.TextContent]:
    """
    Process and integrate data into the knowledge graph.

    ⚠️ **SYNCHRONOUS OPERATION**: This function blocks until processing completes.
    Processing time depends on dataset size (typically 10 seconds to 5 minutes).

    Use cognify_status() to check progress if you need to monitor a long-running process.

    Parameters:
    - datasets: List of knowledge base IDs to process
    - node_set: Optional list of specific nodes to process (default: all nodes)

    Returns:
        Success message with processing summary
    """
```

**add()** - Already correct, no change needed.

**delete()** - Add similar warning:
```python
"""
Delete data from knowledge bases.

⚠️ **SYNCHRONOUS OPERATION**: Blocks until deletion completes (typically <10 seconds).

Parameters:
- data_id: Specific data item ID to delete
- dataset_name: Delete entire dataset

Either data_id OR dataset_name must be provided (not both).
"""
```

#### Files Modified

| File | Function | Change |
|------|----------|--------|
| tools.py | cognify() | Update docstring (lines 68-81) |
| tools.py | delete() | Update docstring (lines 412-421) |
| tools.py | prune() | Update docstring (lines 792-800) |

**Time:** 1 hour (docstring updates + testing)

---

### Improvement 3: ~~Cache Sentence-Transformers Model~~ ✅ N/A

**Status:** NOT APPLICABLE
**Reason:** LocalReranker removed in "Keep Only Jina" step

This improvement is no longer needed since we removed LocalReranker entirely.

---

### Improvement 4: Fix CORS Middleware

**Severity:** HIGH
**Impact:** POST requests blocked, hardcoded origins ignore config
**Estimated Time:** 1 hour

#### Problem
CORS middleware applied incorrectly:
1. Hardcoded origins instead of using `Settings.CORS_ALLOWED_ORIGINS`
2. Missing POST method in allowed methods
3. Passing middleware instance instead of class

**Location:** `cognee-mcp/src/transport.py:16-29, 42-44`

#### Current Code (SSE Transport)

**Lines 16-29:**
```python
async def run_sse_transport(mcp: FastMCP, host: str, port: int, log_level: str):
    """Run the SSE transport for the MCP server."""
    sse_app = mcp.sse_app()

    # ❌ WRONG: Hardcoded origins, missing POST, wrong middleware application
    cors_middleware = CORSMiddleware(
        app=sse_app,
        allow_origins=["http://localhost:3000"],  # ❌ Hardcoded
        allow_credentials=True,
        allow_methods=["GET"],  # ❌ Missing POST
        allow_headers=["*"],
    )

    config = Config(
        app=cors_middleware,  # ❌ Passing instance instead of using add_middleware
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
```

#### Fixed Code

```python
from config import Settings  # ✅ Import Settings

async def run_sse_transport(mcp: FastMCP, host: str, port: int, log_level: str):
    """Run the SSE transport for the MCP server."""
    sse_app = mcp.sse_app()

    # ✅ CORRECT: Use Settings, include POST, proper middleware application
    sse_app.add_middleware(
        CORSMiddleware,  # ✅ Pass class, not instance
        allow_origins=Settings.CORS_ALLOWED_ORIGINS,  # ✅ Use config
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],  # ✅ Include POST
        allow_headers=["Content-Type", "Authorization"],  # ✅ Specific headers
    )

    config = Config(
        app=sse_app,  # ✅ Pass app directly, middleware already applied
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
```

#### Current Code (HTTP Transport)

**Lines 42-44:**
```python
async def run_http_transport(mcp: FastMCP, host: str, port: int, log_level: str):
    """Run the HTTP transport for the MCP server."""
    http_app = mcp.http_app()

    # Same CORS issues as SSE
```

#### Fixed Code

```python
async def run_http_transport(mcp: FastMCP, host: str, port: int, log_level: str):
    """Run the HTTP transport for the MCP server."""
    http_app = mcp.http_app()

    # ✅ Apply CORS middleware correctly
    http_app.add_middleware(
        CORSMiddleware,
        allow_origins=Settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    config = Config(app=http_app, host=host, port=port, log_level=log_level.lower())
```

#### Verify Settings Configuration

**Check:** `cognee-mcp/src/config.py`

```python
# Should have CORS settings defined
CORS_ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
```

If missing, add to Settings class.

#### Testing

```bash
# Test CORS preflight (OPTIONS request)
curl -X OPTIONS http://127.0.0.1:8000/sse \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -v

# Should return:
# Access-Control-Allow-Origin: http://localhost:3000
# Access-Control-Allow-Methods: GET, POST, OPTIONS

# Test POST request works
curl -X POST http://127.0.0.1:8000/sse \
  -H "Origin: http://localhost:3000" \
  -H "Content-Type: application/json" \
  -v

# Should return:
# Access-Control-Allow-Origin: http://localhost:3000
```

**File Modified:** `cognee-mcp/src/transport.py` (lines 16-29, 42-55)

---

### Improvement 5: Add datasetId Support

**Severity:** MEDIUM
**Impact:** Feature parity with backend API
**Estimated Time:** 2 hours

#### Problem
Backend supports both `dataset_id` (single UUID) and `dataset_ids` (list), but MCP only supports `datasets` (list of names).

**Location:** `cognee-mcp/src/cognee_client.py`, `cognee-mcp/src/tools.py`

#### Backend API Endpoints

**From review:**
```python
# Backend accepts:
POST /api/v1/search
{
  "query_text": "search query",
  "dataset_id": "uuid-string",  # Single dataset by ID
  # OR
  "dataset_ids": ["uuid1", "uuid2"],  # Multiple datasets by ID
  # OR
  "datasets": ["name1", "name2"]  # Multiple datasets by name
}
```

#### Current MCP Implementation

**search() tool** (tools.py:491-501):
```python
async def search(
    search_query: str,
    search_type: Literal[...] = "GRAPH_COMPLETION",
    datasets: Optional[List[str]] = None,  # ✅ Names only
    # ❌ MISSING: dataset_id, dataset_ids
    top_k: int = 10,
    ...
)
```

#### Proposed Enhancement

**Option A: Add separate parameters** (Recommended for clarity):
```python
async def search(
    search_query: str,
    search_type: Literal[...] = "GRAPH_COMPLETION",
    datasets: Optional[List[str]] = None,  # KB names
    dataset_id: Optional[str] = None,  # Single KB by UUID
    dataset_ids: Optional[List[str]] = None,  # Multiple KBs by UUID
    top_k: int = 10,
    ...
) -> List[types.TextContent]:
    """
    Search knowledge bases by name or UUID.

    Parameters:
    - datasets: List of KB names (e.g., ["adhd_knowledge", "it_architecture"])
    - dataset_id: Single KB UUID (e.g., "123e4567-e89b-12d3-a456-426614174000")
    - dataset_ids: List of KB UUIDs

    ⚠️ Use ONLY ONE of: datasets, dataset_id, or dataset_ids (not multiple)
    """
    # Validation
    provided = sum([
        datasets is not None,
        dataset_id is not None,
        dataset_ids is not None
    ])

    if provided > 1:
        raise ValueError("Provide only one of: datasets, dataset_id, or dataset_ids")
```

**Option B: Keep simple** (Current approach):
- Only support `datasets` (names)
- Document that UUIDs are not supported via MCP
- Users manage name→UUID mapping in frontend

**Recommendation:** **Option B** - Keep it simple. UUIDs are internal implementation detail, names are user-facing.

**Decision:** SKIP this improvement. Current `datasets` (names) approach is sufficient for user's use case.

---

### Phase 2 Summary

| Improvement | File | Lines Changed | Time | Status |
|-------------|------|---------------|------|--------|
| 1. Optional/Literal types | tools.py | ~15 | 2 hours | ⏳ TODO |
| 2. Document synchronous | tools.py | ~30 | 1 hour | ⏳ TODO |
| 3. ~~Cache model~~ | N/A | N/A | N/A | ✅ N/A |
| 4. Fix CORS | transport.py | ~10 | 1 hour | ⏳ TODO |
| 5. ~~Add datasetId~~ | N/A | N/A | N/A | ✅ SKIP |

**Total Time:** ~4 hours (down from 8-10 hours due to simplifications)

---

## Phase 3: Code Quality & Security

**Priority:** MEDIUM
**Estimated Time:** 2-3 days
**Can Complete After:** Initial production deployment

### Overview

4 code quality and security improvements:
1. Replace global variables with dependency injection
2. Fix configuration validation
3. Security fixes (CORS injection, unsafe class loading)
4. Split tools.py into focused modules (optional)

---

### Improvement 1: Replace Global Variables with DI

**Severity:** MEDIUM
**Impact:** Better testability, thread safety
**Estimated Time:** 1 day

#### Problem
Global mutable variables without thread safety:
- `cognee_client` instance (tools.py)
- `logger` instances (all modules)
- Configuration singletons

**Location:** `cognee-mcp/src/tools.py`, multiple modules

#### Current Pattern

**tools.py:**
```python
# ❌ Global mutable state
_cognee_client = None

def get_cognee_client() -> CogneeClient:
    """Get or create the global Cognee client."""
    global _cognee_client
    if _cognee_client is None:
        _cognee_client = CogneeClient(...)
    return _cognee_client
```

#### Proposed Pattern: Dependency Injection

**Create dependency container** (`cognee-mcp/src/dependencies.py`):
```python
from typing import Optional
from cognee_client import CogneeClient
from config import Settings
import logging

class DependencyContainer:
    """
    Dependency injection container for MCP server.

    Manages lifecycle of shared resources:
    - HTTP client
    - Configuration
    - Logger instances
    """

    def __init__(self):
        self._cognee_client: Optional[CogneeClient] = None
        self._settings = Settings()
        self._logger = logging.getLogger(__name__)

    @property
    def cognee_client(self) -> CogneeClient:
        """Get or create Cognee client (lazy initialization)."""
        if self._cognee_client is None:
            self._cognee_client = CogneeClient(
                use_api=self._settings.USE_API,
                api_url=self._settings.API_URL,
            )
        return self._cognee_client

    @property
    def settings(self) -> Settings:
        """Get settings instance."""
        return self._settings

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        if self._cognee_client:
            await self._cognee_client.close()
            self._cognee_client = None

# Global container instance (created once at startup)
container = DependencyContainer()
```

**Update tools.py:**
```python
from dependencies import container

@mcp.tool()
async def search(...) -> List[types.TextContent]:
    """Search knowledge bases."""
    # ✅ Use dependency container instead of global
    cognee_client = container.cognee_client

    async def search_task(...):
        search_results = await cognee_client.search(...)
        # ...
```

**Update server.py:**
```python
from dependencies import container

async def main():
    """Main entry point."""
    # ... initialization ...

    try:
        if args.transport == "sse":
            await transport.run_sse_transport(mcp, args.host, args.port, args.log_level)
        # ...
    finally:
        # ✅ Cleanup on shutdown
        await container.cleanup()
```

#### Benefits
- ✅ Testable (inject mock dependencies)
- ✅ Clear resource lifecycle
- ✅ Centralized configuration
- ✅ Thread-safe (container manages instances)

#### Testing
```python
# test_tools.py
from dependencies import DependencyContainer
from unittest.mock import AsyncMock

async def test_search_with_mock():
    # Create test container with mock client
    test_container = DependencyContainer()
    test_container._cognee_client = AsyncMock()
    test_container._cognee_client.search.return_value = [{"text": "test"}]

    # Inject test container
    import tools
    tools.container = test_container

    # Test search
    result = await tools.search("test query")
    assert "test" in str(result)
```

**Files Created/Modified:**
- `cognee-mcp/src/dependencies.py` (NEW, ~80 lines)
- `cognee-mcp/src/tools.py` (update imports, remove global variables)
- `cognee-mcp/src/server.py` (add cleanup call)

**Time:** 1 day

---

### Improvement 2: Fix Configuration Validation

**Severity:** MEDIUM
**Impact:** Catch configuration errors early
**Estimated Time:** 0.5 day

#### Problem
`validate_settings()` creates `errors` list but never populates it, always returns True.

**Location:** `cognee-mcp/src/config.py:104-131`

#### Current Code
```python
def validate_settings() -> bool:
    """
    Validate required settings for the MCP server.

    Returns:
        True if settings are valid, False otherwise
    """
    errors = []  # ❌ Never appended to!

    if Settings.USE_API:
        # Check API configuration
        if not Settings.API_URL:
            logger.error("API_URL is required when USE_API=true")
        if not Settings.API_AUTH_TOKEN:
            logger.warning("API_AUTH_TOKEN not set - using no authentication")

    # ... more checks that only log, never append to errors ...

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    return True  # ❌ Always True!
```

#### Fixed Code
```python
def validate_settings() -> bool:
    """
    Validate required settings for the MCP server.

    Returns:
        True if settings are valid, False otherwise

    Raises:
        ValueError: If critical settings are missing
    """
    errors = []
    warnings = []

    # API Mode Validation
    if Settings.USE_API:
        if not Settings.API_URL:
            errors.append("API_URL is required when USE_API=true")  # ✅ Append error

        if not Settings.API_AUTH_TOKEN:
            warnings.append("API_AUTH_TOKEN not set - using no authentication")

    # Reranking Validation
    if Settings.ENABLE_RERANKING:
        jina_api_key = os.getenv("JINA_API_KEY")
        if not jina_api_key:
            errors.append(
                "JINA_API_KEY required for reranking. "
                "Get free key: https://jina.ai/?sui=apikey"
            )

    # CORS Validation
    if not Settings.CORS_ALLOWED_ORIGINS:
        warnings.append("CORS_ALLOWED_ORIGINS not set - will block cross-origin requests")

    # Log warnings (non-fatal)
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Fail on errors
    if errors:
        logger.error("Configuration validation FAILED:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError("Invalid configuration - see errors above")  # ✅ Fail fast

    logger.info("✅ Configuration validation passed")
    return True
```

#### Testing
```bash
# Test validation failure
USE_API=true API_URL="" python cognee-mcp/src/server.py

# Should output:
# ERROR: Configuration validation FAILED:
#   - API_URL is required when USE_API=true
# ValueError: Invalid configuration - see errors above

# Test validation success
USE_API=true API_URL="http://localhost:8000" python cognee-mcp/src/server.py

# Should output:
# INFO: ✅ Configuration validation passed
```

**File Modified:** `cognee-mcp/src/config.py:104-131`
**Time:** 4 hours

---

### Improvement 3: Security Fixes

**Severity:** MEDIUM
**Impact:** Prevent CORS injection, unsafe class loading
**Estimated Time:** 0.5 day

#### Security Issue 1: CORS Origin Injection

**Problem:** If `CORS_ALLOWED_ORIGINS` comes from untrusted input, attacker can inject malicious origins.

**Location:** `cognee-mcp/src/config.py`

**Current:**
```python
# .env file (user-editable)
CORS_ALLOWED_ORIGINS=http://localhost:3000

# config.py
CORS_ALLOWED_ORIGINS: List[str] = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
```

**Attack:**
```bash
# Attacker sets malicious origin
export CORS_ALLOWED_ORIGINS="http://localhost:3000,http://evil.com"
```

**Fix: Validate origins**
```python
def validate_cors_origins(origins: List[str]) -> List[str]:
    """
    Validate CORS origins to prevent injection attacks.

    Args:
        origins: List of origin URLs

    Returns:
        Validated origins

    Raises:
        ValueError: If invalid origin detected
    """
    import re

    validated = []
    origin_pattern = re.compile(r'^https?://[a-zA-Z0-9\-\.]+:[0-9]{1,5}$|^https?://[a-zA-Z0-9\-\.]+$')

    for origin in origins:
        origin = origin.strip()
        if not origin:
            continue

        if not origin_pattern.match(origin):
            raise ValueError(f"Invalid CORS origin: {origin}")

        validated.append(origin)

    return validated

# In Settings class
CORS_ALLOWED_ORIGINS: List[str] = validate_cors_origins(
    os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
)
```

#### Security Issue 2: Unsafe Class Loading

**Problem:** If class names come from untrusted input, attacker can load malicious classes.

**Location:** Not found in current code, but common pattern in Python.

**Prevention:** If you ever dynamically load classes, use allowlist:
```python
# BAD (if class_name is from user input)
cls = globals()[class_name]()

# GOOD
ALLOWED_CLASSES = {
    "JinaReranker": JinaReranker,
    # ... explicit allowlist
}

def safe_load_class(class_name: str):
    if class_name not in ALLOWED_CLASSES:
        raise ValueError(f"Class not allowed: {class_name}")
    return ALLOWED_CLASSES[class_name]
```

#### Security Issue 3: Missing Input Validation

**Add validation for user inputs:**
```python
# tools.py - validate search parameters
async def search(
    search_query: str,
    search_type: Literal[...] = "GRAPH_COMPLETION",
    datasets: Optional[List[str]] = None,
    top_k: int = 10,
    ...
):
    """Search knowledge bases."""
    # ✅ Validate inputs
    if not search_query or len(search_query) > 10000:
        raise ValueError("search_query must be 1-10000 characters")

    if top_k < 1 or top_k > 100:
        raise ValueError("top_k must be between 1 and 100")

    if datasets:
        for dataset in datasets:
            if not re.match(r'^[a-zA-Z0-9_\-]+$', dataset):
                raise ValueError(f"Invalid dataset name: {dataset}")

    # ... continue with search
```

**Files Modified:**
- `cognee-mcp/src/config.py` (add CORS validation)
- `cognee-mcp/src/tools.py` (add input validation)

**Time:** 4 hours

---

### Improvement 4: Split tools.py into Modules (Optional)

**Severity:** LOW
**Impact:** Better maintainability
**Estimated Time:** 1 day

#### Problem
`tools.py` is 1,044 lines - still large after LocalReranker removal.

#### Proposed Structure

```
cognee-mcp/src/
├── tools/
│   ├── __init__.py          # Re-export all tools
│   ├── cognify_tools.py     # cognify(), cognify_status()
│   ├── search_tools.py      # search(), list_datasets()
│   ├── data_tools.py        # add(), delete(), prune()
│   └── dev_tools.py         # save_interaction(), get_developer_rules()
```

#### Implementation Example

**tools/cognify_tools.py:**
```python
"""Cognee cognify (processing) tools."""

import asyncio
from typing import List, Optional
import mcp.types as types
from dependencies import container
from cognee.shared.logging_utils import get_logger

logger = get_logger()

@mcp.tool()
async def cognify(
    datasets: List[str],
    node_set: Optional[List[str]] = None,
) -> List[types.TextContent]:
    """Process and integrate data into the knowledge graph."""
    # ... implementation ...

@mcp.tool()
async def cognify_status(
    dataset_name: Optional[str] = None,
) -> List[types.TextContent]:
    """Check cognify processing status."""
    # ... implementation ...
```

**tools/__init__.py:**
```python
"""Cognee MCP tools."""

# Re-export all tools
from .cognify_tools import cognify, cognify_status
from .search_tools import search, list_datasets
from .data_tools import add, delete, prune
from .dev_tools import save_interaction, get_developer_rules

__all__ = [
    "cognify",
    "cognify_status",
    "search",
    "list_datasets",
    "add",
    "delete",
    "prune",
    "save_interaction",
    "get_developer_rules",
]
```

**server.py:**
```python
# Import from tools package
from tools import (
    cognify,
    cognify_status,
    search,
    list_datasets,
    add,
    delete,
    prune,
    save_interaction,
    get_developer_rules,
)

# Or just import tools package if using decorators
import tools  # All @mcp.tool() decorators auto-register
```

#### Benefits
- ✅ Easier to find specific tools
- ✅ Smaller files (<300 lines each)
- ✅ Logical grouping by functionality
- ✅ Easier to test individual modules

#### Drawback
- ⚠️ More files to navigate
- ⚠️ Import complexity increases slightly

**Recommendation:** OPTIONAL - Current 1,044 lines is manageable. Only split if you plan to add many more tools (>15 total).

**Time:** 1 day (if pursued)

---

### Phase 3 Summary

| Improvement | File(s) | Lines Changed | Time | Priority |
|-------------|---------|---------------|------|----------|
| 1. Dependency injection | dependencies.py, tools.py, server.py | +100 | 1 day | MEDIUM |
| 2. Config validation | config.py | ~30 | 0.5 day | MEDIUM |
| 3. Security fixes | config.py, tools.py | ~50 | 0.5 day | MEDIUM |
| 4. Split tools.py | tools/*.py | 0 (refactor) | 1 day | LOW |

**Total Time:** 2-3 days
**Recommended:** Do items 1-3, skip item 4

---

## Testing Strategy

### Unit Tests

**Create:** `cognee-mcp/tests/test_tools.py`

```python
import pytest
from unittest.mock import AsyncMock, patch
from tools import search, cognify, add, delete
from dependencies import DependencyContainer

@pytest.fixture
def mock_container():
    """Create mock dependency container."""
    container = DependencyContainer()
    container._cognee_client = AsyncMock()
    return container

@pytest.mark.asyncio
async def test_search_with_jina_reranker(mock_container):
    """Test search with Jina reranker."""
    # Mock search results
    mock_container.cognee_client.search.return_value = [
        {"text": "Result 1", "score": 0.9},
        {"text": "Result 2", "score": 0.8},
    ]

    # Mock Jina reranker
    with patch('reranker.JinaReranker') as mock_reranker:
        mock_reranker_instance = AsyncMock()
        mock_reranker_instance.rerank.return_value = [
            {"index": 1, "relevance_score": 0.95, "document": "Result 2"},
            {"index": 0, "relevance_score": 0.85, "document": "Result 1"},
        ]
        mock_reranker.return_value = mock_reranker_instance

        # Test search
        result = await search(
            search_query="test query",
            rerank=True,
            rerank_provider="jina"
        )

        # Verify reranking applied
        assert "Result 2" in str(result)  # Higher ranked result first

@pytest.mark.asyncio
async def test_search_with_rerank_off(mock_container):
    """Test search with reranking disabled."""
    mock_container.cognee_client.search.return_value = [
        {"text": "Result 1", "score": 0.9},
    ]

    result = await search(
        search_query="test query",
        rerank_provider="off"
    )

    assert "Result 1" in str(result)
    # Verify no reranking applied
    mock_container.cognee_client.search.assert_called_once()

@pytest.mark.asyncio
async def test_cognify_with_datasets(mock_container):
    """Test cognify with dataset list."""
    mock_container.cognee_client.cognify.return_value = "Success"

    result = await cognify(datasets=["adhd_knowledge", "it_architecture"])

    assert "Success" in str(result)
    mock_container.cognee_client.cognify.assert_called_once_with(
        datasets=["adhd_knowledge", "it_architecture"],
        node_set=None
    )
```

**Run tests:**
```bash
pytest cognee-mcp/tests/test_tools.py -v
```

---

### Integration Tests

**Create:** `cognee-mcp/tests/integration/test_mcp_server.py`

```python
import asyncio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@pytest.mark.asyncio
async def test_mcp_server_stdio():
    """Test MCP server via stdio transport."""
    server_params = StdioServerParameters(
        command="python",
        args=["cognee-mcp/src/server.py", "--transport", "stdio"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            assert len(tools) >= 9, "Should have at least 9 tools"

            # Test search tool
            result = await session.call_tool(
                "search",
                arguments={
                    "search_query": "test query",
                    "rerank_provider": "off"
                }
            )

            assert result is not None

@pytest.mark.asyncio
async def test_search_with_backend():
    """Test search with actual backend (requires backend running)."""
    # This test requires cognee backend running at localhost:8000
    pytest.skip("Requires backend running - manual test only")

    # Start MCP server
    # ... test with real backend ...
```

**Run integration tests:**
```bash
# Requires backend running
pytest cognee-mcp/tests/integration/ -v
```

---

### Manual Testing Checklist

```markdown
## Manual Testing Checklist

### Setup
- [ ] Backend API running (`uvicorn cognee.api.server:app`)
- [ ] `.env` file configured with JINA_API_KEY
- [ ] MCP server running (`python cognee-mcp/src/server.py --transport sse`)

### Test Cases

#### Search Functionality
- [ ] Search with rerank_provider="jina" returns ranked results
- [ ] Search with rerank_provider="off" returns backend order
- [ ] Search with datasets=["kb1", "kb2"] searches multiple KBs
- [ ] Search with datasets=None searches all KBs
- [ ] Invalid rerank_provider raises clear error

#### Cognify Functionality
- [ ] Cognify with datasets=["kb1"] processes single KB
- [ ] Cognify with datasets=["kb1", "kb2"] processes multiple KBs
- [ ] Cognify status shows progress
- [ ] Cognify with invalid dataset name fails gracefully

#### Data Management
- [ ] Add text data to KB works
- [ ] Delete by data_id works
- [ ] Delete by dataset_name removes entire KB
- [ ] Prune removes processed data
- [ ] List datasets returns all KBs

#### Transport Modes
- [ ] SSE transport accepts connections
- [ ] HTTP transport accepts connections
- [ ] Stdio transport doesn't corrupt protocol stream
- [ ] CORS allows requests from allowed origins
- [ ] CORS blocks requests from disallowed origins

#### Error Handling
- [ ] Missing JINA_API_KEY shows clear error
- [ ] Backend offline shows clear error
- [ ] Invalid parameters show validation errors
- [ ] Network errors are caught and logged
```

---

## Rollback Plan

### If Critical Bugs Found During Implementation

**Rollback to Previous Version:**

```bash
# Commit before starting
git add .
git commit -m "Pre-implementation checkpoint"
git tag pre-implementation-$(date +%Y%m%d)

# If rollback needed
git reset --hard pre-implementation-$(date +%Y%m%d)
```

### If LocalReranker Removal Causes Issues

**Temporarily Re-enable LocalReranker:**

1. Revert `reranker.py` changes:
   ```bash
   git checkout HEAD -- cognee-mcp/src/reranker.py
   ```

2. Revert `tools.py` changes:
   ```bash
   git checkout HEAD -- cognee-mcp/src/tools.py
   ```

3. Add context manager to LocalReranker (quick fix):
   ```python
   class LocalReranker:
       async def __aenter__(self):
           return self

       async def __aexit__(self, exc_type, exc_val, exc_tb):
           pass
   ```

### If Phase 1 Fixes Break Functionality

**Rollback Individual Fixes:**

Each fix is isolated, so you can rollback individual changes:

```bash
# Rollback stdout redirect
git checkout HEAD -- cognee-mcp/src/server.py

# Rollback HTTP client fix
git checkout HEAD -- cognee-mcp/src/cognee_client.py

# Rollback return types
git checkout HEAD -- cognee-mcp/src/tools.py
```

---

## Implementation Checklist

### Pre-Implementation
- [ ] Create backup branch: `git checkout -b backup-$(date +%Y%m%d)`
- [ ] Commit current state: `git commit -m "Pre-implementation checkpoint"`
- [ ] Run existing tests: `pytest cognee-mcp/tests/ -v`
- [ ] Document current behavior for comparison

### Keep Only Jina (1-2 hours)
- [ ] Remove LocalReranker class from reranker.py (lines 165-230)
- [ ] Update RerankerFactory in reranker.py (lines 239-262)
- [ ] Remove local reranker block from tools.py (lines 613-653)
- [ ] Update search() docstring in tools.py (line 517)
- [ ] Verify compilation: `python3 -m py_compile cognee-mcp/src/*.py`
- [ ] Test search with Jina: `search(query="test", rerank_provider="jina")`
- [ ] Test search with off: `search(query="test", rerank_provider="off")`
- [ ] Commit: `git commit -m "Remove LocalReranker, keep only Jina"`

### Phase 1: Critical Bugs (3 hours)
- [ ] ~~Fix LocalReranker context manager~~ (N/A - removed)
- [ ] Fix stdio stdout redirect (server.py:85)
- [ ] Test stdio transport: `python server.py --transport stdio`
- [ ] Fix HTTP client leak (cognee_client.py:89-98)
- [ ] Test HTTP client cleanup
- [ ] Fix return type annotations (tools.py - all functions)
- [ ] Run type checker: `python3 -m mypy cognee-mcp/src/tools.py`
- [ ] Fix API parameter format (cognee_client.py:136)
- [ ] Test API mode with node_set=None
- [ ] Commit: `git commit -m "Phase 1: Fix critical bugs"`

### Phase 2: High Priority (4 hours)
- [ ] Add Optional/Literal type hints (tools.py)
- [ ] Run type checker: `python3 -m mypy --strict`
- [ ] Update docstrings for synchronous operations (tools.py)
- [ ] ~~Cache sentence-transformers model~~ (N/A - removed)
- [ ] Fix CORS middleware (transport.py)
- [ ] Test CORS with curl
- [ ] ~~Add datasetId support~~ (SKIP - not needed)
- [ ] Commit: `git commit -m "Phase 2: High-priority improvements"`

### Phase 3: Code Quality (2-3 days)
- [ ] Create dependencies.py with DI container
- [ ] Update tools.py to use container
- [ ] Update server.py with cleanup
- [ ] Fix configuration validation (config.py)
- [ ] Test validation with invalid config
- [ ] Add security fixes (CORS validation, input validation)
- [ ] ~~Split tools.py into modules~~ (OPTIONAL - skip for now)
- [ ] Commit: `git commit -m "Phase 3: Code quality improvements"`

### Testing
- [ ] Write unit tests (test_tools.py)
- [ ] Run unit tests: `pytest -v`
- [ ] Write integration tests (test_mcp_server.py)
- [ ] Run integration tests with backend
- [ ] Manual testing checklist (see above)
- [ ] Performance testing (search latency, reranking time)

### Documentation
- [ ] Update README.md with new features
- [ ] Update DOCS/implementation_summary.md
- [ ] Document removed features (LocalReranker)
- [ ] Add migration guide for users

### Deployment
- [ ] Build Docker image: `docker build -t cognee/cognee-mcp:latest .`
- [ ] Test Docker deployment
- [ ] Update docker-compose.yml
- [ ] Deploy to staging environment
- [ ] Run smoke tests in staging
- [ ] Deploy to production

---

## Success Criteria

### After "Keep Only Jina"
- ✅ LocalReranker class removed (100+ lines)
- ✅ Reranker tests pass
- ✅ Search with Jina works
- ✅ Search with off works
- ✅ Attempting "local" raises clear error

### After Phase 1
- ✅ All 5 critical bugs fixed
- ✅ Stdio transport works without corruption
- ✅ No resource leaks
- ✅ Type checker passes
- ✅ API mode works correctly

### After Phase 2
- ✅ Strict type checking passes
- ✅ Clear documentation for synchronous operations
- ✅ CORS works for allowed origins
- ✅ Performance within acceptable range

### After Phase 3
- ✅ Dependency injection implemented
- ✅ Configuration validation works
- ✅ Security vulnerabilities fixed
- ✅ Code quality metrics improved

---

## Time Estimates Summary

| Phase | Tasks | Estimated Time | Priority |
|-------|-------|----------------|----------|
| **Keep Only Jina** | Remove LocalReranker | 1-2 hours | MEDIUM |
| **Phase 1** | 4 critical bugs | 3 hours | CRITICAL |
| **Phase 2** | 3 high-priority improvements | 4 hours | HIGH |
| **Phase 3** | 3 code quality improvements | 2-3 days | MEDIUM |
| **Testing** | Unit + integration tests | 1 day | HIGH |
| **Documentation** | Update docs | 4 hours | MEDIUM |
| **TOTAL** | All phases + testing + docs | **3-4 days (24-32 hours)** | |

---

## Conclusion

This implementation plan provides:

1. **Clear Step-by-Step Instructions** - Exact file:line references for every change
2. **Rationale for Each Decision** - Why we're removing LocalReranker, why each bug matters
3. **Testing Strategy** - Unit tests, integration tests, manual testing checklist
4. **Rollback Plan** - How to recover if something goes wrong
5. **Success Criteria** - Clear metrics for completion

**Recommended Execution Order:**
1. ✅ Keep Only Jina (1-2 hours) - Simplifies codebase, removes 2 bugs
2. ✅ Phase 1 (3 hours) - Fix remaining critical bugs
3. ✅ Phase 2 (4 hours) - High-priority improvements
4. ⏸️ Phase 3 (2-3 days) - Can be done after initial deployment

**Total Critical Path Time:** ~8 hours for production-ready MCP server

After completing Keep Only Jina + Phase 1 + Phase 2, you'll have a solid, bug-free, protocol-compliant MCP server ready for production use with your private knowledge bases.
