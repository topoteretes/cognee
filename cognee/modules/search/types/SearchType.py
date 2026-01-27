from enum import Enum


class SearchType(str, Enum):
    SUMMARIES = "SUMMARIES"  # TODO: Update with search refactor
    CHUNKS = "CHUNKS"  # TODO: Update with search refactor
    RAG_COMPLETION = "RAG_COMPLETION"
    TRIPLET_COMPLETION = "TRIPLET_COMPLETION"  # TODO: Update with search refactor
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    GRAPH_SUMMARY_COMPLETION = "GRAPH_SUMMARY_COMPLETION"  # TODO: Update with search refactor
    CYPHER = "CYPHER"  # TODO: Update with search refactor
    NATURAL_LANGUAGE = "NATURAL_LANGUAGE"  # TODO: Update with search refactor
    GRAPH_COMPLETION_COT = "GRAPH_COMPLETION_COT"
    GRAPH_COMPLETION_CONTEXT_EXTENSION = "GRAPH_COMPLETION_CONTEXT_EXTENSION"
    FEELING_LUCKY = "FEELING_LUCKY"
    TEMPORAL = "TEMPORAL"  # TODO: Update with search refactor
    CODING_RULES = "CODING_RULES"  # TODO: Update with search refactor
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"  # TODO: Update with search refactor
