# High-Priority MCP Fixes - Implementation Plan

**Date:** 2025-11-12
**Total Estimated Effort:** 15-18 hours
**Critical Requirement:** Transparent authentication - NO manual token handling for MCP end users

---

## Overview

This plan addresses 3 critical issues identified in the architectural reviews to make the Cognee MCP server production-ready:

1. **Resource Management** - HTTP client leaks
2. **Async Error Handling** - Silent task failures
3. **Code Organization** - 1,233-line monolithic server.py

---

## Task 1: Wrap HTTP Clients in Context Managers
**Priority:** Critical | **Effort:** 4-5 hours | **Risk:** Medium

### Problem
```python
# Current: Resource leaks
self.client = httpx.AsyncClient(timeout=300.0)  # No cleanup
async with aiohttp.ClientSession() as session:  # New session per request
```

### Solution
```python
# New: Proper async context managers
class CogneeClient:
    @asynccontextmanager
    async def get_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300.0)
        try:
            yield self._client
        finally:
            pass  # Reused, closed in close()

    async def close(self):
        if self._client:
            await self._client.aclose()

class JinaReranker:
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(...)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
```

### Files Modified
- ✅ `cognee-mcp/src/reranker.py` (NEW)
- ✅ `cognee-mcp/src/cognee_client.py` (UPDATE)
- ✅ `cognee-mcp/src/server.py` (UPDATE usage)

---

## Task 2: Replace asyncio.create_task() with TaskGroup
**Priority:** Critical | **Effort:** 3-4 hours | **Risk:** Medium

### Problem
```python
# Current: Fire-and-forget, silent failures
asyncio.create_task(cognify_task(...))  # No error tracking
```

### Solution
```python
# New: Structured concurrency with error propagation
async def cognify(data: str) -> list:
    try:
        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(cognify_task(...))
            # Automatic error propagation
    except ExceptionGroup as eg:
        logger.error(f"Cognify failed: {eg}")
        raise ValueError(f"Failed: {eg}") from eg
```

### Pattern Changes
- **cognify()**: Single background task → TaskGroup
- **cognify_add_developer_rules()**: Multiple parallel tasks → TaskGroup
- **save_interaction()**: Background task → TaskGroup
- **codify()**: Background task → TaskGroup

### Benefits
✅ Exceptions propagate immediately
✅ No silent failures
✅ Parent cancellation cancels children
✅ Better debugging

---

## Task 3: Split server.py into Modules
**Priority:** High | **Effort:** 6-8 hours | **Risk:** High

### Current State
```
server.py: 1,233 lines
├── JinaReranker class
├── @mcp.tool() functions (11 tools)
├── Transport setup (SSE/HTTP/stdio)
└── Main entry point
```

### Target Structure
```
cognee-mcp/src/
├── server.py        (~100 lines)  - Main entry point
├── tools.py         (~500 lines)  - All @mcp.tool() functions
├── reranker.py      (~150 lines)  - JinaReranker + LocalReranker
├── transport.py     (~150 lines)  - SSE/HTTP setup
├── config.py        (~100 lines)  - Settings, environment
├── health.py        (~100 lines)  - Health checks
└── cognee_client.py (existing)
```

### Module Breakdown

#### config.py (NEW)
```python
"""Configuration with transparent auth"""
class Settings:
    JINA_API_KEY: Optional[str] = os.getenv("JINA_API_KEY")
    CORS_ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]
    # No user-facing auth - all transparent
```

#### reranker.py (NEW)
```python
"""Reranking with proper resource management"""
class JinaReranker:
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...

class LocalReranker:
    """Sentence-transformers fallback"""
    @staticmethod
    async def rerank(query, documents, top_n): ...
```

#### tools.py (NEW)
```python
"""All MCP tools with TaskGroup patterns"""
async def search(...) -> list:
    async with asyncio.TaskGroup() as tg:
        result = tg.create_task(search_task(...))
    return [types.TextContent(type="text", text=result.result())]

# All 11 tools extracted here
```

#### transport.py (NEW)
```python
"""Transport setup with configurable CORS"""
def setup_sse_transport(mcp: FastMCP) -> callable:
    sse_app = mcp.sse_app()
    sse_app.add_middleware(create_cors_middleware())
    return run_sse

def create_cors_middleware():
    return Middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        ...
    )
```

#### health.py (NEW)
```python
"""Health checks with dependency status"""
async def detailed_health_check():
    return JSONResponse({
        "status": "ok",
        "checks": {
            "jina_api": "configured" if settings.JINA_API_KEY else "not_configured"
        }
    })
```

#### server.py (REFACTORED)
```python
"""Main entry point - delegates to modules"""
mcp = FastMMC("Cognee", instructions=SERVER_INSTRUCTIONS)
setup_health_routes(mcp)

# Initialize and inject dependencies
cognee_client = CogneeClient(api_url=..., api_token=...)
set_cognee_client(cognee_client)

# Select transport and run
if args.transport == "sse":
    run_sse = setup_sse_transport(mcp)
    await run_sse()
```

---

## Implementation Timeline

### Week 1 (8-10 hours)
- **Day 1-2**: Task 1 - HTTP context managers
- **Day 3**: Task 2 - TaskGroup migration
- **Day 4**: Testing

### Week 2 (6-8 hours)
- **Day 5-6**: Task 3 - Module split
- **Day 7**: Integration testing

### Parallel Execution
- ✅ Task 1 and 2 can run in parallel
- ⚠️ Task 3 depends on tasks 1 and 2

---

## Testing Strategy

### Task 1: Resource Management
```python
@pytest.mark.asyncio
async def test_jina_reranker_cleanup():
    reranker = JinaReranker("test")
    async with reranker as r:
        assert r._session is not None
    assert r._session is None  # Verified cleanup

@pytest.mark.asyncio
async def test_no_resource_leaks():
    # Run many operations
    # Check with leak detection
```

### Task 2: TaskGroup Error Handling
```python
@pytest.mark.asyncio
async def test_errors_propagate():
    with pytest.raises(ValueError):
        await cognify("invalid_data")  # Error must propagate

@pytest.mark.asyncio
async def test_partial_failure():
    # Create scenario with some task failures
    # Verify successful tasks complete
```

### Task 3: Module Structure
```python
# Test each module independently
# Test integration
# Test backward compatibility
# Verify server.py < 150 lines
```

---

## Authentication Design (Transparent to End Users)

### Operator Configuration (One-Time Setup)
```bash
# .env file - server operator only
JINA_API_KEY=sk-xxx                    # Server-to-Jina API
BACKEND_API_TOKEN=internal-secret      # Server-to-Cognee backend
CORS_ALLOWED_ORIGINS=http://localhost:3000,https://example.com
```

### MCP Client Usage (Zero Configuration)
```json
// LibreChat or any MCP client - NO TOKENS
{
  "mcpServers": {
    "cognee": {
      "type": "sse",
      "url": "http://server:8000/sse"
      // NO AUTH - handled by server internally
    }
  }
}
```

### Token Management (Server-Side Only)
- Credentials stored in environment variables
- Used automatically by server
- No exposure to MCP protocol
- Rotated by operators, not users

---

## Risk Assessment

| Risk | Task 1 | Task 2 | Task 3 |
|------|--------|--------|--------|
| Breaking Changes | Low | Low | High |
| Resource Leaks | Fixed | N/A | N/A |
| Silent Failures | N/A | Fixed | N/A |
| Code Complexity | Decreases | Decreases | Decreases |
| Test Coverage | Needs new tests | Needs new tests | Full re-test |

### Mitigation Strategies
1. **Git tags** at each milestone
2. **Comprehensive tests** before and after
3. **Incremental migration** (one module at a time)
4. **Rollback procedures** documented
5. **Backward compatibility** preserved where possible

---

## Success Criteria

### Task 1: HTTP Context Managers
- [ ] No resource leaks (verified with leak detection tools)
- [ ] All HTTP clients properly closed
- [ ] No performance regression
- [ ] All existing tests pass + new tests

### Task 2: TaskGroup Migration
- [ ] All async errors properly propagate
- [ ] No silent failures in logs
- [ ] Parent cancellation works correctly
- [ ] All existing tests pass + new tests

### Task 3: Module Split
- [ ] server.py reduced to <150 lines
- [ ] All functionality preserved
- [ ] Clear separation of concerns
- [ ] Easier to test and maintain
- [ ] All tests pass in new structure

---

## Benefits After Implementation

### Code Quality
✅ **No resource leaks** - Proper cleanup
✅ **Visible errors** - No silent failures
✅ **Modular structure** - Easier to understand
✅ **Better testability** - Single responsibility

### Production Readiness
✅ **Resource efficiency** - No leaks
✅ **Reliability** - Errors visible
✅ **Maintainability** - Modular code
✅ **Scalability** - Better async patterns

### Developer Experience
✅ **Easier debugging** - Structured concurrency
✅ **Faster development** - Modular structure
✅ **Better testing** - Isolated components
✅ **Clear documentation** - Each module documented

---

## Next Steps

1. **Review Plan** - Approve or request changes
2. **Start Implementation** - Begin with Task 1
3. **Test After Each Task** - Verify success criteria
4. **Document Lessons** - Update implementation guide
5. **Deploy** - Follow migration checklist

---

**Estimated Total Time:** 15-18 hours over 1-2 weeks
**Complexity:** Medium-High (Task 3 is complex refactoring)
**Risk:** Medium (mitigated with testing and rollback)
**ROI:** High (eliminates critical production issues)

---

## Files to Modify/Create

### New Files
- `cognee-mcp/src/config.py` - Configuration
- `cognee-mcp/src/reranker.py` - Reranking logic
- `cognee-mcp/src/transport.py` - Transport setup
- `cognee-mcp/src/health.py` - Health checks
- `cognee-mcp/src/tools.py` - All MCP tools

### Modified Files
- `cognee-mcp/src/cognee_client.py` - Add context manager
- `cognee-mcp/src/server.py` - Refactor to delegate

### Deleted Code
- JinaReranker class (moved to reranker.py)
- @mcp.tool() functions (moved to tools.py)
- Transport setup (moved to transport.py)
- Health routes (moved to health.py)

---

**Ready to proceed with implementation?**