"""Data models for the cognitive architecture."""

from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Set

from pydantic import BaseModel, Field
from cognee.infrastructure.llm.config import get_llm_config

if get_llm_config().llm_provider.lower() == "gemini":
    """
    Note: Gemini doesn't allow for an empty dictionary to be a part of the data model
    so we created new data models to bypass that issue, but other LLMs have slightly worse performance
    when creating knowledge graphs with these data models compared to the old data models
    so now there's an if statement here so that the rest of the LLMs can use the old data models.
    """

    class Node(BaseModel):
        """Node in a knowledge graph."""

        id: str
        name: str
        type: str
        description: str
        label: str
        layer_id: Optional[str] = None  # New field to track which layer this node belongs to

    class Edge(BaseModel):
        """Edge in a knowledge graph."""

        source_node_id: str
        target_node_id: str
        relationship_name: str
        layer_id: Optional[str] = None  # New field to track which layer this edge belongs to

    class KnowledgeGraph(BaseModel):
        """Knowledge graph."""

        summary: str
        description: str
        nodes: List[Node] = Field(..., default_factory=list)
        edges: List[Edge] = Field(..., default_factory=list)
else:
    class Node(BaseModel):
        """Node in a knowledge graph."""

        id: str
        name: str
        type: str
        description: str
        layer_id: Optional[str] = None  # New field to track which layer this node belongs to
        properties: Optional[Dict[str, Any]] = Field(
            default_factory=dict, description="Node properties"
        )

    class Edge(BaseModel):
        """Edge in a knowledge graph."""

        source_node_id: str
        target_node_id: str
        relationship_name: str
        layer_id: Optional[str] = None  # New field to track which layer this edge belongs to
        properties: Optional[Dict[str, Any]] = Field(
            default_factory=dict, description="Edge properties"
        )

    class KnowledgeGraph(BaseModel):
        """Knowledge graph."""

        nodes: List[Node] = Field(..., default_factory=list)
        edges: List[Edge] = Field(..., default_factory=list)
        name: str = "Knowledge Graph"
        description: str = ""


class Layer(BaseModel):
    """Layer in a layered knowledge graph."""
    
    id: str
    name: str
    description: str
    layer_type: str = "default"  # Type of layer (e.g., "base", "enrichment", "inference")
    parent_layers: List[str] = Field(default_factory=list)  # IDs of parent layers this layer builds upon
    properties: Optional[Dict[str, Any]] = None


class LayeredKnowledgeGraph(BaseModel):
    """Knowledge graph with explicit support for layers."""
    
    # Core graph data
    base_graph: KnowledgeGraph
    
    # Layer management
    layers: List[Layer] = Field(..., default_factory=list)
    
    # Metadata
    name: str = "Layered Knowledge Graph"
    description: str = ""
    
    def get_layer_graph(self, layer_id: str) -> KnowledgeGraph:
        """
        Get a subgraph containing only nodes and edges from the specified layer.
        
        Args:
            layer_id: The ID of the layer to extract
            
        Returns:
            A knowledge graph containing only the nodes and edges from that layer
        """
        # Filter nodes and edges that belong to this layer
        nodes = [node for node in self.base_graph.nodes if node.layer_id == layer_id]
        edges = [edge for edge in self.base_graph.edges if edge.layer_id == layer_id]
        
        # Check if nodes and edges have name and description fields
        has_name = hasattr(self.base_graph, 'name')
        has_description = hasattr(self.base_graph, 'description')
        
        # Create a new KnowledgeGraph with only the layer's nodes and edges
        if has_name and has_description:
            # Find the layer to get its name
            layer = next((layer for layer in self.layers if layer.id == layer_id), None)
            layer_name = layer.name if layer else f"Layer {layer_id}"
            
            return KnowledgeGraph(
                nodes=nodes,
                edges=edges,
                name=f"{layer_name} Graph",
                description=f"Subgraph containing only nodes and edges from {layer_name}"
            )
        else:
            # For Gemini-compatible model
            return KnowledgeGraph(
                nodes=nodes,
                edges=edges,
                summary=f"Layer {layer_id} Graph",
                description=f"Subgraph containing only nodes and edges from layer {layer_id}"
            )
    
    def get_cumulative_layer_graph(self, layer_id: str) -> KnowledgeGraph:
        """
        Get a subgraph containing nodes and edges from the specified layer and all its parent layers.
        
        Args:
            layer_id: The ID of the layer to extract (and its parents)
            
        Returns:
            A knowledge graph containing the cumulative nodes and edges
        """
        # Collect all layer IDs that should be included
        layer_ids = set()
        self._collect_parent_layers(layer_id, layer_ids)
        
        # Filter nodes and edges that belong to any of the collected layers
        nodes = [node for node in self.base_graph.nodes if node.layer_id in layer_ids]
        edges = [edge for edge in self.base_graph.edges if edge.layer_id in layer_ids]
        
        # Check if nodes and edges have name and description fields
        has_name = hasattr(self.base_graph, 'name')
        has_description = hasattr(self.base_graph, 'description')
        
        # Create a new KnowledgeGraph with the cumulative nodes and edges
        if has_name and has_description:
            # Find the layer to get its name
            layer = next((layer for layer in self.layers if layer.id == layer_id), None)
            layer_name = layer.name if layer else f"Layer {layer_id}"
            
            return KnowledgeGraph(
                nodes=nodes,
                edges=edges,
                name=f"{layer_name} Cumulative Graph",
                description=f"Cumulative subgraph containing nodes and edges from {layer_name} and its parent layers"
            )
        else:
            # For Gemini-compatible model
            return KnowledgeGraph(
                nodes=nodes,
                edges=edges,
                summary=f"Layer {layer_id} Cumulative Graph",
                description=f"Cumulative subgraph containing nodes and edges from layer {layer_id} and its parent layers"
            )
    
    def _collect_parent_layers(self, layer_id: str, collected_layer_ids: Set[str]) -> None:
        """
        Recursively collect the given layer ID and all its parent layer IDs.
        
        Args:
            layer_id: The ID of the layer to start from
            collected_layer_ids: Set to collect the layer IDs into
        """
        collected_layer_ids.add(layer_id)
        
        # Find the layer object
        layer = next((layer for layer in self.layers if layer.id == layer_id), None)
        if layer:
            # Add all its parent layers and their parents
            for parent_id in layer.parent_layers:
                if parent_id not in collected_layer_ids:
                    self._collect_parent_layers(parent_id, collected_layer_ids)
    
    def add_layer(self, layer: Layer) -> None:
        """
        Add a layer to the layered graph.
        
        Args:
            layer: The layer to add
        """
        # Check if parent layers exist
        for parent_id in layer.parent_layers:
            if not any(existing_layer.id == parent_id for existing_layer in self.layers):
                raise ValueError(f"Parent layer with ID {parent_id} does not exist")
        
        self.layers.append(layer)
    
    def add_node_to_layer(self, node: Node, layer_id: str) -> None:
        """
        Add a node to a specific layer.
        
        Args:
            node: The node to add
            layer_id: The ID of the layer to add the node to
        """
        # Check if the layer exists
        if not any(layer.id == layer_id for layer in self.layers):
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        
        # Set the layer ID on the node
        node_copy = node.copy()
        node_copy.layer_id = layer_id
        
        # Add to the base graph
        self.base_graph.nodes.append(node_copy)
    
    def add_edge_to_layer(self, edge: Edge, layer_id: str) -> None:
        """
        Add an edge to a specific layer.
        
        Args:
            edge: The edge to add
            layer_id: The ID of the layer to add the edge to
        """
        # Check if the layer exists
        if not any(layer.id == layer_id for layer in self.layers):
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        
        # Set the layer ID on the edge
        edge_copy = edge.copy()
        edge_copy.layer_id = layer_id
        
        # Add to the base graph
        self.base_graph.edges.append(edge_copy)


class GraphQLQuery(BaseModel):
    """GraphQL query."""

    query: str


class Answer(BaseModel):
    """Answer."""

    answer: str


class ChunkStrategy(Enum):
    EXACT = "exact"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    CODE = "code"
    LANGCHAIN_CHARACTER = "langchain_character"


class ChunkEngine(Enum):
    LANGCHAIN_ENGINE = "langchain"
    DEFAULT_ENGINE = "default"
    HAYSTACK_ENGINE = "haystack"


class MemorySummary(BaseModel):
    """Memory summary."""

    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)


class TextSubclass(str, Enum):
    ARTICLES = "Articles, essays, and reports"
    BOOKS = "Books and manuscripts"
    NEWS_STORIES = "News stories and blog posts"
    RESEARCH_PAPERS = "Research papers and academic publications"
    SOCIAL_MEDIA = "Social media posts and comments"
    WEBSITE_CONTENT = "Website content and product descriptions"
    PERSONAL_NARRATIVES = "Personal narratives and stories"
    SPREADSHEETS = "Spreadsheets and tables"
    FORMS = "Forms and surveys"
    DATABASES = "Databases and CSV files"
    SOURCE_CODE = "Source code in various programming languages"
    SHELL_SCRIPTS = "Shell commands and scripts"
    MARKUP_LANGUAGES = "Markup languages (HTML, XML)"
    STYLESHEETS = "Stylesheets (CSS) and configuration files (YAML, JSON, INI)"
    CHAT_TRANSCRIPTS = "Chat transcripts and messaging history"
    CUSTOMER_SERVICE_LOGS = "Customer service logs and interactions"
    CONVERSATIONAL_AI = "Conversational AI training data"
    TEXTBOOK_CONTENT = "Textbook content and lecture notes"
    EXAM_QUESTIONS = "Exam questions and academic exercises"
    E_LEARNING_MATERIALS = "E-learning course materials"
    POETRY = "Poetry and prose"
    SCRIPTS = "Scripts for plays, movies, and television"
    SONG_LYRICS = "Song lyrics"
    MANUALS = "Manuals and user guides"
    TECH_SPECS = "Technical specifications and API documentation"
    HELPDESK_ARTICLES = "Helpdesk articles and FAQs"
    LEGAL_CONTRACTS = "Contracts and agreements"
    LAWS = "Laws, regulations, and legal case documents"
    POLICY_DOCUMENTS = "Policy documents and compliance materials"
    CLINICAL_TRIALS = "Clinical trial reports"
    PATIENT_RECORDS = "Patient records and case notes"
    SCIENTIFIC_ARTICLES = "Scientific journal articles"
    FINANCIAL_REPORTS = "Financial reports and statements"
    BUSINESS_PLANS = "Business plans and proposals"
    MARKET_RESEARCH = "Market research and analysis reports"
    AD_COPIES = "Ad copies and marketing slogans"
    PRODUCT_CATALOGS = "Product catalogs and brochures"
    PRESS_RELEASES = "Press releases and promotional content"
    PROFESSIONAL_EMAILS = "Professional and formal correspondence"
    PERSONAL_EMAILS = "Personal emails and letters"
    IMAGE_CAPTIONS = "Image and video captions"
    ANNOTATIONS = "Annotations and metadata for various media"
    VOCAB_LISTS = "Vocabulary lists and grammar rules"
    LANGUAGE_EXERCISES = "Language exercises and quizzes"
    LEGAL_AND_REGULATORY_DOCUMENTS = "Legal and Regulatory Documents"
    OTHER_TEXT = "Other types of text data"


class AudioSubclass(str, Enum):
    MUSIC_TRACKS = "Music tracks and albums"
    PODCASTS = "Podcasts and radio broadcasts"
    AUDIOBOOKS = "Audiobooks and audio guides"
    INTERVIEWS = "Recorded interviews and speeches"
    SOUND_EFFECTS = "Sound effects and ambient sounds"
    OTHER_AUDIO = "Other types of audio recordings"


class ImageSubclass(str, Enum):
    PHOTOGRAPHS = "Photographs and digital images"
    ILLUSTRATIONS = "Illustrations, diagrams, and charts"
    INFOGRAPHICS = "Infographics and visual data representations"
    ARTWORK = "Artwork and paintings"
    SCREENSHOTS = "Screenshots and graphical user interfaces"
    OTHER_IMAGES = "Other types of images"


class VideoSubclass(str, Enum):
    MOVIES = "Movies and short films"
    DOCUMENTARIES = "Documentaries and educational videos"
    TUTORIALS = "Video tutorials and how-to guides"
    ANIMATED_FEATURES = "Animated features and cartoons"
    LIVE_EVENTS = "Live event recordings and sports broadcasts"
    OTHER_VIDEOS = "Other types of video content"


class MultimediaSubclass(str, Enum):
    WEB_CONTENT = "Interactive web content and games"
    VR_EXPERIENCES = "Virtual reality (VR) and augmented reality (AR) experiences"
    MIXED_MEDIA = "Mixed media presentations and slide decks"
    E_LEARNING_MODULES = "E-learning modules with integrated multimedia"
    DIGITAL_EXHIBITIONS = "Digital exhibitions and virtual tours"
    OTHER_MULTIMEDIA = "Other types of multimedia content"


class Model3DSubclass(str, Enum):
    ARCHITECTURAL_RENDERINGS = "Architectural renderings and building plans"
    PRODUCT_MODELS = "Product design models and prototypes"
    ANIMATIONS = "3D animations and character models"
    SCIENTIFIC_VISUALIZATIONS = "Scientific simulations and visualizations"
    VR_OBJECTS = "Virtual objects for AR/VR applications"
    OTHER_3D_MODELS = "Other types of 3D models"


class ProceduralSubclass(str, Enum):
    TUTORIALS_GUIDES = "Tutorials and step-by-step guides"
    WORKFLOW_DESCRIPTIONS = "Workflow and process descriptions"
    SIMULATIONS = "Simulation and training exercises"
    RECIPES = "Recipes and crafting instructions"
    OTHER_PROCEDURAL = "Other types of procedural content"


class ContentType(BaseModel):
    """Base class for different types of content."""

    type: str


class TextContent(ContentType):
    type: str = "TEXTUAL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    subclass: List[TextSubclass]


class AudioContent(ContentType):
    type: str = "AUDIO_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    subclass: List[AudioSubclass]


class ImageContent(ContentType):
    type: str = "IMAGE_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    subclass: List[ImageSubclass]


class VideoContent(ContentType):
    type: str = "VIDEO_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    subclass: List[VideoSubclass]


class MultimediaContent(ContentType):
    type: str = "MULTIMEDIA_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    subclass: List[MultimediaSubclass]


class Model3DContent(ContentType):
    type: str = "3D_MODEL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    subclass: List[Model3DSubclass]


class ProceduralContent(ContentType):
    type: str = "PROCEDURAL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    subclass: List[ProceduralSubclass]


class DefaultContentPrediction(BaseModel):
    """Class for a single class label prediction."""

    label: Union[
        TextSubclass,
        AudioSubclass,
        ImageSubclass,
        VideoSubclass,
        MultimediaSubclass,
        Model3DSubclass,
        ProceduralSubclass,
    ]
    confidence: float
    content_type: ContentType


class SummarizedContent(BaseModel):
    """Class for a single class label summary and description."""

    summary: str
    description: str


class SummarizedFunction(BaseModel):
    name: str
    description: str
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    decorators: Optional[List[str]] = None


class SummarizedClass(BaseModel):
    name: str
    description: str
    methods: Optional[List[SummarizedFunction]] = None
    decorators: Optional[List[str]] = None


class SummarizedCode(BaseModel):
    high_level_summary: str
    key_features: List[str]
    imports: List[str] = []
    constants: List[str] = []
    classes: List[SummarizedClass] = []
    functions: List[SummarizedFunction] = []
    workflow_description: Optional[str] = None


class GraphDBType(Enum):
    NETWORKX = auto()
    NEO4J = auto()
    FALKORDB = auto()


# Models for representing different entities
class Relationship(BaseModel):
    type: str
    source: Optional[str] = None
    target: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None


class DocumentType(BaseModel):
    type_id: str
    description: str
    default_relationship: Relationship = Relationship(type="is_type")


class Category(BaseModel):
    category_id: str
    name: str
    default_relationship: Relationship = Relationship(type="categorized_as")


class Document(BaseModel):
    id: str
    type: str
    title: str


class UserLocation(BaseModel):
    location_id: str
    description: str
    default_relationship: Relationship = Relationship(type="located_in")


class UserProperties(BaseModel):
    custom_properties: Optional[Dict[str, Any]] = None
    location: Optional[UserLocation] = None


class DefaultGraphModel(BaseModel):
    node_id: str
    user_properties: UserProperties = UserProperties()
    documents: List[Document] = []
    default_fields: Optional[Dict[str, Any]] = {}
    default_relationship: Relationship = Relationship(type="has_properties")


class ChunkSummary(BaseModel):
    text: str
    chunk_id: str


class ChunkSummaries(BaseModel):
    """Relevant summary and chunk id"""

    summaries: List[ChunkSummary]


class MonitoringTool(str, Enum):
    """Monitoring tools"""

    LANGFUSE = "langfuse"
    LLMLITE = "llmlite"
    LANGSMITH = "langsmith"
