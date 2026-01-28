from enum import Enum


class SearchType(str, Enum):
    SUMMARIES = "SUMMARIES"  # TODO: Update with search refactor - Andrej
    CHUNKS = "CHUNKS"  # TODO: Update with search refactor - Andrej
    RAG_COMPLETION = "RAG_COMPLETION"
    TRIPLET_COMPLETION = "TRIPLET_COMPLETION"  # TODO: Update with search refactor - Andrej
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    GRAPH_SUMMARY_COMPLETION = (
        "GRAPH_SUMMARY_COMPLETION"  # TODO: Update with search refactor - Andrej
    )
    CYPHER = "CYPHER"  # TODO: Update with search refactor - Igor
    NATURAL_LANGUAGE = "NATURAL_LANGUAGE"  # TODO: Update with search refactor - Igor
    GRAPH_COMPLETION_COT = "GRAPH_COMPLETION_COT"
    GRAPH_COMPLETION_CONTEXT_EXTENSION = "GRAPH_COMPLETION_CONTEXT_EXTENSION"
    FEELING_LUCKY = "FEELING_LUCKY"
    TEMPORAL = "TEMPORAL"  # Test temporal Igor
    CODING_RULES = "CODING_RULES"  # Test coding rules Igor
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"  # TODO: Update with search refactor - Andrej
