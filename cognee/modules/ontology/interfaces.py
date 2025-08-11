"""Abstract interfaces for ontology system components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Callable
from pydantic import BaseModel
from enum import Enum


class OntologyFormat(str, Enum):
    """Supported ontology formats."""
    RDF_XML = "rdf_xml"
    OWL = "owl"
    JSON = "json"
    CSV = "csv"
    YAML = "yaml"
    DATABASE = "database"
    LLM_GENERATED = "llm_generated"


class OntologyScope(str, Enum):
    """Ontology scopes for different use cases."""
    GLOBAL = "global"  # Applies to all pipelines
    DOMAIN = "domain"  # Applies to specific domain (medical, legal, etc.)
    PIPELINE = "pipeline"  # Applies to specific pipeline type
    USER = "user"  # User-specific ontologies
    DATASET = "dataset"  # Dataset-specific ontologies


class OntologyNode(BaseModel):
    """Standard ontology node representation."""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    category: Optional[str] = None
    properties: Dict[str, Any] = {}
    labels: List[str] = []


class OntologyEdge(BaseModel):
    """Standard ontology edge representation."""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = {}
    weight: Optional[float] = None


class OntologyGraph(BaseModel):
    """Standard ontology graph representation."""
    id: str
    name: str
    description: Optional[str] = None
    format: OntologyFormat
    scope: OntologyScope
    nodes: List[OntologyNode]
    edges: List[OntologyEdge]
    metadata: Dict[str, Any] = {}


class OntologyContext(BaseModel):
    """Context for ontology operations."""
    user_id: Optional[str] = None
    dataset_id: Optional[str] = None
    pipeline_name: Optional[str] = None
    domain: Optional[str] = None
    custom_properties: Dict[str, Any] = {}


class DataPointMapping(BaseModel):
    """Mapping configuration between ontology and DataPoint."""
    ontology_node_type: str
    datapoint_class: str
    field_mappings: Dict[str, str] = {}  # ontology_field -> datapoint_field
    custom_resolver: Optional[str] = None  # Function name for custom resolution
    validation_rules: List[str] = []


class GraphBindingConfig(BaseModel):
    """Configuration for how ontology binds to graph structures."""
    node_type_mapping: Dict[str, str] = {}  # ontology_type -> graph_node_type
    edge_type_mapping: Dict[str, str] = {}  # ontology_relation -> graph_edge_type
    property_transformations: Dict[str, Callable[[Any], Any]] = {}
    custom_binding_strategy: Optional[str] = None


class IOntologyProvider(ABC):
    """Abstract interface for ontology providers."""

    @abstractmethod
    async def load_ontology(
        self, 
        source: Union[str, Dict[str, Any]], 
        context: Optional[OntologyContext] = None
    ) -> OntologyGraph:
        """Load ontology from source."""
        pass

    @abstractmethod
    async def save_ontology(
        self, 
        ontology: OntologyGraph, 
        destination: str,
        context: Optional[OntologyContext] = None
    ) -> bool:
        """Save ontology to destination."""
        pass

    @abstractmethod
    def supports_format(self, format: OntologyFormat) -> bool:
        """Check if provider supports given format."""
        pass

    @abstractmethod
    async def validate_ontology(self, ontology: OntologyGraph) -> bool:
        """Validate ontology structure."""
        pass


class IOntologyAdapter(ABC):
    """Abstract interface for ontology adapters."""

    @abstractmethod
    async def find_matching_nodes(
        self,
        query_text: str,
        ontology: OntologyGraph,
        similarity_threshold: float = 0.8
    ) -> List[OntologyNode]:
        """Find nodes matching query text."""
        pass

    @abstractmethod
    async def get_node_relationships(
        self,
        node_id: str,
        ontology: OntologyGraph,
        max_depth: int = 2
    ) -> List[OntologyEdge]:
        """Get relationships for a specific node."""
        pass

    @abstractmethod
    async def expand_subgraph(
        self,
        node_ids: List[str],
        ontology: OntologyGraph,
        directed: bool = True
    ) -> Tuple[List[OntologyNode], List[OntologyEdge]]:
        """Expand subgraph around given nodes."""
        pass

    @abstractmethod
    async def merge_ontologies(
        self,
        ontologies: List[OntologyGraph]
    ) -> OntologyGraph:
        """Merge multiple ontologies."""
        pass


class IOntologyRegistry(ABC):
    """Abstract interface for ontology registry."""

    @abstractmethod
    async def register_ontology(
        self,
        ontology: OntologyGraph,
        scope: OntologyScope,
        context: Optional[OntologyContext] = None
    ) -> str:
        """Register an ontology."""
        pass

    @abstractmethod
    async def get_ontology(
        self,
        ontology_id: str,
        context: Optional[OntologyContext] = None
    ) -> Optional[OntologyGraph]:
        """Get ontology by ID."""
        pass

    @abstractmethod
    async def find_ontologies(
        self,
        scope: Optional[OntologyScope] = None,
        domain: Optional[str] = None,
        context: Optional[OntologyContext] = None
    ) -> List[OntologyGraph]:
        """Find ontologies matching criteria."""
        pass

    @abstractmethod
    async def unregister_ontology(
        self,
        ontology_id: str,
        context: Optional[OntologyContext] = None
    ) -> bool:
        """Unregister an ontology."""
        pass


class IDataPointResolver(ABC):
    """Abstract interface for resolving ontology to DataPoint instances."""

    @abstractmethod
    async def resolve_to_datapoint(
        self,
        ontology_node: OntologyNode,
        mapping_config: DataPointMapping,
        context: Optional[OntologyContext] = None
    ) -> Any:  # Should be DataPoint but avoiding circular import
        """Resolve ontology node to DataPoint instance."""
        pass

    @abstractmethod
    async def resolve_from_datapoint(
        self,
        datapoint: Any,  # DataPoint
        mapping_config: DataPointMapping,
        context: Optional[OntologyContext] = None
    ) -> OntologyNode:
        """Resolve DataPoint instance to ontology node."""
        pass

    @abstractmethod
    async def validate_mapping(
        self,
        mapping_config: DataPointMapping
    ) -> bool:
        """Validate mapping configuration."""
        pass

    @abstractmethod
    def register_custom_resolver(
        self,
        resolver_name: str,
        resolver_func: Callable[[OntologyNode, DataPointMapping], Any]
    ) -> None:
        """Register a custom resolver function."""
        pass


class IGraphBinder(ABC):
    """Abstract interface for binding ontology to graph structures."""

    @abstractmethod
    async def bind_ontology_to_graph(
        self,
        ontology: OntologyGraph,
        binding_config: GraphBindingConfig,
        context: Optional[OntologyContext] = None
    ) -> Tuple[List[Any], List[Any]]:  # (graph_nodes, graph_edges)
        """Bind ontology to graph structure."""
        pass

    @abstractmethod
    async def transform_node_properties(
        self,
        node: OntologyNode,
        transformations: Dict[str, Callable[[Any], Any]]
    ) -> Dict[str, Any]:
        """Transform node properties according to binding config."""
        pass

    @abstractmethod
    async def transform_edge_properties(
        self,
        edge: OntologyEdge,
        transformations: Dict[str, Callable[[Any], Any]]
    ) -> Dict[str, Any]:
        """Transform edge properties according to binding config."""
        pass

    @abstractmethod
    def register_binding_strategy(
        self,
        strategy_name: str,
        strategy_func: Callable[[OntologyGraph, GraphBindingConfig], Tuple[List[Any], List[Any]]]
    ) -> None:
        """Register a custom binding strategy."""
        pass


class IOntologyManager(ABC):
    """Abstract interface for ontology manager."""

    @abstractmethod
    async def get_applicable_ontologies(
        self,
        context: OntologyContext
    ) -> List[OntologyGraph]:
        """Get ontologies applicable to given context."""
        pass

    @abstractmethod
    async def enhance_with_ontology(
        self,
        content: str,
        context: OntologyContext
    ) -> Dict[str, Any]:
        """Enhance content with ontological information."""
        pass

    @abstractmethod
    async def inject_ontology_into_task(
        self,
        task_name: str,
        task_params: Dict[str, Any],
        context: OntologyContext
    ) -> Dict[str, Any]:
        """Inject ontological context into task parameters."""
        pass

    @abstractmethod
    async def resolve_to_datapoints(
        self,
        ontology_nodes: List[OntologyNode],
        context: OntologyContext
    ) -> List[Any]:  # List[DataPoint]
        """Resolve ontology nodes to DataPoint instances."""
        pass

    @abstractmethod
    async def bind_to_graph(
        self,
        ontology: OntologyGraph,
        context: OntologyContext
    ) -> Tuple[List[Any], List[Any]]:  # (graph_nodes, graph_edges)
        """Bind ontology to graph structure using configured binding."""
        pass

    @abstractmethod
    def configure_datapoint_mapping(
        self,
        domain: str,
        mappings: List[DataPointMapping]
    ) -> None:
        """Configure DataPoint mappings for a domain."""
        pass

    @abstractmethod
    def configure_graph_binding(
        self,
        domain: str,
        binding_config: GraphBindingConfig
    ) -> None:
        """Configure graph binding for a domain."""
        pass
