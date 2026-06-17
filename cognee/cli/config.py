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
    "datasets": "Manage datasets (list, create, inspect, status, delete)",
    "agents": "Manage agents (create, list, get, delete, register, unregister, connections)",
    "sessions": "View conversation sessions and Q&A history",
    "feedback": "Add or remove feedback on session Q&A entries",
    "memify": "Run the memory enrichment pipeline on a dataset",
    "remember": "Ingest data and build the knowledge graph in a single call",
    "recall": "Search the knowledge graph for relevant information",
    "improve": "Enrich an existing knowledge graph with additional context and rules",
    "forget": "Remove data from the knowledge graph",
    "serve": "Connect to a Cognee instance (cloud or local)",
    "upgrade": "Apply pending relational + data migrations (alembic-style; head or a revision)",
    "downgrade": "Revert data migrations to a revision ('base' or a slug); rewrites data",
    "stamp": "Set the stored migration revision WITHOUT running migrations (bookkeeping repair)",
    "history": "List the data-migration chain, newest first",
    "current": "Show each database's stamped migration revision (and last failure, if any)",
    "push": "Upload a local dataset's knowledge graph to Cognee Cloud",
}

# Search type choices
SEARCH_TYPE_CHOICES = [
    "GRAPH_COMPLETION",
    "RAG_COMPLETION",
    "CHUNKS",
    "SUMMARIES",
    "CODE",
    "CYPHER",
]

# Chunker choices
CHUNKER_CHOICES = ["TextChunker", "LangchainChunker", "CsvChunker"]

# Output format choices
OUTPUT_FORMAT_CHOICES = ["json", "pretty", "simple"]
