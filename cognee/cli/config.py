"""
CLI configuration and constants to avoid hardcoded values
"""

# CLI Constants
CLI_DESCRIPTION = "Cognee CLI - Manage your knowledge graphs and cognitive processing pipelines."
DEFAULT_DOCS_URL = "https://docs.cognee.ai"

# Command descriptions - these should match the actual command implementations
COMMAND_DESCRIPTIONS = {
    "add": "Add data to Cognee for knowledge graph processing",
    "search": "Search and query the knowledge graph for insights, information, and connections",
    "cognify": "Transform ingested data into a structured knowledge graph",
    "delete": "Delete data from cognee knowledge base",
    "config": "Manage cognee configuration settings",
}

# Search type choices
SEARCH_TYPE_CHOICES = [
    "GRAPH_COMPLETION",
    "RAG_COMPLETION",
    "INSIGHTS",
    "CHUNKS",
    "SUMMARIES",
    "CODE",
    "CYPHER",
]

# Chunker choices
CHUNKER_CHOICES = ["TextChunker", "LangchainChunker"]

# Output format choices
OUTPUT_FORMAT_CHOICES = ["json", "pretty", "simple"]
