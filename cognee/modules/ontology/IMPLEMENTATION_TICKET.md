# 🎫 Epic: Refactor Ontology System for General Pipeline Usage

**Epic ID:** CGNEE-2024-ONT-001  
**Priority:** High  
**Story Points:** 21  
**Type:** Epic  
**Labels:** `refactoring`, `architecture`, `ontology`, `pipeline-integration`

## 📋 Overview

Refactor the current monolithic `OntologyResolver` to follow Cognee's architectural patterns and be general, extensible, and usable across all pipelines. The current system is tightly coupled to specific tasks and only supports RDF/OWL formats. We need a modular system that follows Cognee's established patterns for configuration, methods organization, and module structure.

## 🎯 Business Value

- **Consistency**: Follow established Cognee patterns and conventions
- **Flexibility**: Support multiple ontology formats and domains
- **Reusability**: One ontology system usable across all pipelines
- **Maintainability**: Modular architecture following Cognee's separation of concerns
- **Developer Experience**: Familiar patterns and simple configuration

## 🏗️ Architecture Analysis

Based on examination of existing Cognee patterns:

### **Configuration Pattern** (Following `cognee/base_config.py`, `cognee/modules/cognify/config.py`)
```python
# Follow BaseSettings pattern with @lru_cache
class OntologyConfig(BaseSettings):
    default_format: OntologyFormat = OntologyFormat.JSON
    enable_semantic_search: bool = False
    registry_type: str = "memory"
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

@lru_cache
def get_ontology_config():
    return OntologyConfig()
```

### **Methods Organization** (Following `cognee/modules/data/methods/`, `cognee/modules/users/methods/`)
```python
# Organize methods in separate files:
cognee/modules/ontology/methods/
├── __init__.py           # Export all methods
├── create_ontology.py    # async def create_ontology(...)
├── get_ontology.py       # async def get_ontology(...)
├── load_ontology.py      # async def load_ontology(...)
├── register_ontology.py  # async def register_ontology(...)
└── delete_ontology.py    # async def delete_ontology(...)
```

### **Models Structure** (Following `cognee/modules/data/models/`, `cognee/modules/users/models/`)
```python
# Create models with proper inheritance:
cognee/modules/ontology/models/
├── __init__.py                    # Export all models
├── OntologyGraph.py              # class OntologyGraph(BaseModel)
├── OntologyNode.py               # class OntologyNode(BaseModel)
├── OntologyContext.py            # class OntologyContext(BaseModel)
└── DataPointMapping.py           # class DataPointMapping(BaseModel)
```

### **Module Organization** (Following existing module patterns)
```python
cognee/modules/ontology/
├── __init__.py                   # Public API with convenience functions
├── config.py                     # OntologyConfig with @lru_cache
├── models/                       # Pydantic models
├── methods/                      # Async method functions
├── providers/                    # Format-specific providers
├── adapters/                     # Query and search operations
├── operations/                   # Core business logic
└── utils/                        # Utility functions
```

## 📦 Epic Breakdown (Cognee-Style Implementation)

### Story 1: Configuration System & Models
**Story Points:** 3  
**Assignee:** Backend Developer  

#### Acceptance Criteria
- [ ] Create `OntologyConfig` following `BaseSettings` pattern with `@lru_cache`
- [ ] Create Pydantic models in `models/` directory with proper `__init__.py` exports
- [ ] Follow naming conventions: `OntologyNode`, `OntologyEdge`, etc.
- [ ] Use proper type hints and docstring patterns from existing code
- [ ] Environment variable support following existing config patterns

#### Files to Create
```
cognee/modules/ontology/config.py
cognee/modules/ontology/models/__init__.py
cognee/modules/ontology/models/OntologyGraph.py
cognee/modules/ontology/models/OntologyNode.py
cognee/modules/ontology/models/OntologyContext.py
cognee/modules/ontology/models/DataPointMapping.py
```

#### Implementation Notes
```python
# config.py - Follow existing pattern
@lru_cache
def get_ontology_config():
    return OntologyConfig()

# models/OntologyNode.py - Follow DataPoint pattern
class OntologyNode(BaseModel):
    id: str = Field(..., description="Unique identifier")
    name: str
    type: str
    # ... follow existing model patterns
```

---

### Story 2: Methods Organization
**Story Points:** 2  
**Assignee:** Backend Developer  
**Dependencies:** Story 1

#### Acceptance Criteria
- [ ] Create `methods/` directory following existing pattern
- [ ] Implement async functions following `create_dataset`, `get_user` patterns
- [ ] Use proper error handling patterns from existing methods
- [ ] Follow parameter naming and return type conventions
- [ ] Export all methods in `methods/__init__.py`

#### Files to Create
```
cognee/modules/ontology/methods/__init__.py
cognee/modules/ontology/methods/create_ontology.py
cognee/modules/ontology/methods/get_ontology.py
cognee/modules/ontology/methods/load_ontology.py
cognee/modules/ontology/methods/register_ontology.py
cognee/modules/ontology/methods/delete_ontology.py
```

#### Implementation Notes
```python
# methods/create_ontology.py - Follow create_dataset pattern
async def create_ontology(
    ontology_data: Dict[str, Any],
    user: User,
    scope: OntologyScope = OntologyScope.USER
) -> OntologyGraph:
    # Follow existing error handling and validation patterns
```

---

### Story 3: Provider System (Following Existing Patterns)
**Story Points:** 4  
**Assignee:** Backend Developer  
**Dependencies:** Story 2

#### Acceptance Criteria
- [ ] Create providers following the adapter pattern seen in retrieval systems
- [ ] Use abstract base classes like `CogneeAbstractGraph`
- [ ] Follow error handling patterns from existing providers
- [ ] Support graceful degradation (like RDF provider with optional rdflib)
- [ ] Use proper logging patterns with `get_logger()`

#### Files to Create
```
cognee/modules/ontology/providers/__init__.py
cognee/modules/ontology/providers/base.py
cognee/modules/ontology/providers/rdf_provider.py
cognee/modules/ontology/providers/json_provider.py
cognee/modules/ontology/providers/csv_provider.py
```

#### Implementation Notes
```python
# providers/base.py - Follow CogneeAbstractGraph pattern
class BaseOntologyProvider(ABC):
    @abstractmethod
    async def load_ontology(self, source: str) -> OntologyGraph:
        pass

# providers/rdf_provider.py - Follow graceful fallback pattern
class RDFOntologyProvider(BaseOntologyProvider):
    def __init__(self):
        try:
            import rdflib
            self.available = True
        except ImportError:
            logger.warning("rdflib not available")
            self.available = False
```

---

### Story 4: Operations Layer (Core Business Logic)
**Story Points:** 4  
**Assignee:** Senior Backend Developer  
**Dependencies:** Story 3

#### Acceptance Criteria
- [ ] Create `operations/` directory following pipeline operations pattern
- [ ] Implement core business logic following `cognee_pipeline` pattern
- [ ] Use dependency injection patterns seen in existing operations
- [ ] Follow async/await patterns consistently
- [ ] Implement proper error handling and logging

#### Files to Create
```
cognee/modules/ontology/operations/__init__.py
cognee/modules/ontology/operations/ontology_manager.py
cognee/modules/ontology/operations/datapoint_resolver.py
cognee/modules/ontology/operations/graph_binder.py
cognee/modules/ontology/operations/registry.py
```

#### Implementation Notes
```python
# operations/ontology_manager.py - Follow cognee_pipeline pattern
async def manage_ontology_processing(
    context: OntologyContext,
    providers: Dict[str, BaseOntologyProvider],
    config: OntologyConfig = None
) -> List[OntologyGraph]:
    # Follow existing operation patterns
```

---

### Story 5: Pipeline Integration (Following Task Pattern)
**Story Points:** 3  
**Assignee:** Backend Developer  
**Dependencies:** Story 4

#### Acceptance Criteria
- [ ] Create integration following `Task` class pattern
- [ ] Support injection into existing pipeline operations
- [ ] Follow parameter passing patterns from `cognee_pipeline`
- [ ] Maintain backward compatibility with existing tasks
- [ ] Use context variables pattern for configuration

#### Files to Create
```
cognee/modules/ontology/operations/pipeline_integration.py
cognee/modules/ontology/operations/task_enhancer.py
```

#### Implementation Notes
```python
# operations/pipeline_integration.py - Follow cognee_pipeline pattern
async def inject_ontology_context(
    tasks: list[Task],
    ontology_context: OntologyContext,
    config: OntologyConfig = None
) -> list[Task]:
    # Follow existing task enhancement patterns
```

---

### Story 6: Utils and Utilities
**Story Points:** 2  
**Assignee:** Backend Developer  
**Dependencies:** Story 5

#### Acceptance Criteria
- [ ] Create `utils/` directory for utility functions
- [ ] Follow utility patterns from existing modules
- [ ] Implement helper functions for common operations
- [ ] Use proper type hints and error handling

#### Files to Create
```
cognee/modules/ontology/utils/__init__.py
cognee/modules/ontology/utils/ontology_helpers.py
cognee/modules/ontology/utils/mapping_helpers.py
cognee/modules/ontology/utils/validation.py
```

---

### Story 7: Public API and Module Initialization
**Story Points:** 2  
**Assignee:** Backend Developer  
**Dependencies:** Story 6

#### Acceptance Criteria
- [ ] Create comprehensive `__init__.py` following existing patterns
- [ ] Export key classes and functions following module conventions
- [ ] Create convenience functions following `get_base_config()` pattern
- [ ] Maintain backward compatibility with `OntologyResolver`
- [ ] Use proper `__all__` exports

#### Files to Update/Create
```
cognee/modules/ontology/__init__.py
```

#### Implementation Notes
```python
# __init__.py - Follow existing module export patterns
from .config import get_ontology_config
from .models import OntologyGraph, OntologyNode, OntologyContext
from .methods import create_ontology, load_ontology, get_ontology

# Convenience functions following get_base_config pattern
@lru_cache
def get_ontology_manager():
    return OntologyManager()

__all__ = [
    "OntologyGraph",
    "OntologyNode", 
    "get_ontology_config",
    "create_ontology",
    # ... follow existing export patterns
]
```

---

### Story 8: Enhanced Task Implementation
**Story Points:** 2  
**Assignee:** Backend Developer  
**Dependencies:** Story 7

#### Acceptance Criteria
- [ ] Update existing graph extraction task to be ontology-aware
- [ ] Follow existing task parameter patterns
- [ ] Maintain backward compatibility
- [ ] Use proper error handling and fallback mechanisms
- [ ] Follow existing task documentation patterns

#### Files to Create/Update
```
cognee/tasks/graph/extract_graph_from_data_ontology_aware.py
```

---

### Story 9: Documentation and Examples
**Story Points:** 1  
**Assignee:** Technical Writer / Backend Developer  
**Dependencies:** Story 8

#### Acceptance Criteria
- [ ] Create migration guide following existing documentation patterns
- [ ] Provide examples following existing code patterns
- [ ] Document configuration options following existing config docs
- [ ] Create troubleshooting guide

#### Files to Create
```
cognee/modules/ontology/MIGRATION_GUIDE.md
cognee/modules/ontology/examples/
```

---

### Story 10: Testing Following Cognee Patterns
**Story Points:** 2  
**Assignee:** QA Engineer / Backend Developer  
**Dependencies:** Story 9

#### Acceptance Criteria
- [ ] Create tests following existing test structure in `cognee/tests/`
- [ ] Use existing test patterns and fixtures
- [ ] Test integration with existing pipeline system
- [ ] Verify backward compatibility
- [ ] Performance testing following existing benchmarks

#### Files to Create
```
cognee/tests/unit/ontology/
cognee/tests/integration/ontology/
```

---

## 🔧 Technical Requirements (Cognee-Aligned)

### Follow Existing Patterns
- **Configuration**: Use `BaseSettings` with `@lru_cache` pattern
- **Models**: Pydantic models with proper inheritance and validation
- **Methods**: Async functions with consistent error handling
- **Operations**: Business logic separation following existing operations
- **Exports**: Proper `__init__.py` files with `__all__` exports
- **Logging**: Use `get_logger()` pattern consistently
- **Error Handling**: Follow existing exception patterns

### Integration Requirements
- Work seamlessly with existing `Task` system
- Support existing `cognee_pipeline` operations
- Integrate with current configuration management
- Support existing database patterns
- Maintain compatibility with current DataPoint model

### Code Style Requirements
- Follow existing naming conventions (PascalCase for classes, snake_case for functions)
- Use type hints consistently like existing code
- Follow docstring patterns from existing modules
- Use existing import organization patterns
- Follow async/await patterns consistently

## 🚀 Implementation Guidelines

### File Organization (Following Cognee Patterns)
```
cognee/modules/ontology/
├── __init__.py                    # Public API, convenience functions
├── config.py                      # OntologyConfig with @lru_cache  
├── models/                        # Pydantic models
│   ├── __init__.py               # Export all models
│   ├── OntologyGraph.py
│   ├── OntologyNode.py
│   └── ...
├── methods/                       # Business methods (CRUD operations)
│   ├── __init__.py               # Export all methods
│   ├── create_ontology.py
│   ├── get_ontology.py
│   └── ...
├── providers/                     # Format-specific providers
├── operations/                    # Core business logic
├── utils/                         # Utility functions
└── examples/                      # Usage examples
```

### Code Patterns to Follow

#### Configuration Pattern
```python
# cognee/modules/ontology/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class OntologyConfig(BaseSettings):
    default_format: str = "json"
    enable_semantic_search: bool = False
    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    
    def to_dict(self) -> dict:
        return {"default_format": self.default_format, ...}

@lru_cache
def get_ontology_config():
    return OntologyConfig()
```

#### Method Pattern
```python
# cognee/modules/ontology/methods/create_ontology.py
from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.models import OntologyGraph

logger = get_logger("ontology.create")

async def create_ontology(
    ontology_data: Dict[str, Any],
    user: User,
    scope: OntologyScope = OntologyScope.USER
) -> OntologyGraph:
    """Create ontology following existing method patterns."""
    try:
        # Implementation following existing patterns
        logger.info(f"Creating ontology for user: {user.id}")
        # ...
    except Exception as e:
        logger.error(f"Failed to create ontology: {e}")
        raise
```

#### Model Pattern
```python
# cognee/modules/ontology/models/OntologyNode.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class OntologyNode(BaseModel):
    """Ontology node following Cognee model patterns."""
    
    id: str = Field(..., description="Unique identifier")
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Follow existing model config patterns."""
        arbitrary_types_allowed = True
```

### Integration with Existing Systems

#### Task Integration
```python
# Follow existing Task parameter patterns
async def extract_graph_from_data_ontology_aware(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel] = KnowledgeGraph,
    ontology_config: OntologyConfig = None,
    **kwargs
) -> List[DocumentChunk]:
    """Enhanced task following existing task patterns."""
    
    config = ontology_config or get_ontology_config()
    # Follow existing task implementation patterns
```

#### Pipeline Integration
```python
# Follow cognee_pipeline pattern for integration
async def cognee_pipeline_with_ontology(
    tasks: list[Task],
    ontology_context: OntologyContext = None,
    **kwargs
):
    """Enhanced pipeline following existing pipeline patterns."""
    
    # Inject ontology context following existing parameter injection
    enhanced_tasks = []
    for task in tasks:
        if ontology_context:
            # Enhance task following existing enhancement patterns
            enhanced_task = enhance_task_with_ontology(task, ontology_context)
            enhanced_tasks.append(enhanced_task)
        else:
            enhanced_tasks.append(task)
    
    # Use existing pipeline execution
    return await cognee_pipeline(enhanced_tasks, **kwargs)
```

## 📊 Success Metrics (Aligned with Cognee Standards)

### Code Quality Metrics
- [ ] Follow all existing linting and code style rules
- [ ] Pass all existing code quality checks  
- [ ] Maintain or improve test coverage percentage
- [ ] Follow existing documentation standards

### Integration Metrics
- [ ] Zero breaking changes to existing API
- [ ] All existing tests continue to pass
- [ ] Performance meets or exceeds existing benchmarks
- [ ] Memory usage within existing parameters

### Pattern Compliance
- [ ] Configuration follows `BaseSettings` + `@lru_cache` pattern
- [ ] Models follow existing Pydantic patterns
- [ ] Methods follow existing async function patterns
- [ ] Exports follow existing `__init__.py` patterns
- [ ] Error handling follows existing exception patterns

## 🔗 Related Files to Study

### Configuration Patterns
- `cognee/base_config.py`
- `cognee/modules/cognify/config.py`
- `cognee/infrastructure/llm/config.py`

### Model Patterns  
- `cognee/modules/data/models/`
- `cognee/modules/users/models/`
- `cognee/infrastructure/engine/models/DataPoint.py`

### Method Patterns
- `cognee/modules/data/methods/`
- `cognee/modules/users/methods/`

### Operation Patterns
- `cognee/modules/pipelines/operations/`
- `cognee/modules/search/methods/`

### Module Organization
- `cognee/modules/pipelines/__init__.py`
- `cognee/modules/data/__init__.py`
- `cognee/modules/users/__init__.py`

---

**Estimated Total Effort:** 21 Story Points (~4-5 Sprints)  
**Target Completion:** End of Q2 2024  
**Review Required:** Architecture Review, Code Standards Review, Integration Review

**Key Success Factor:** Strict adherence to existing Cognee patterns and conventions to ensure seamless integration and maintainability.
