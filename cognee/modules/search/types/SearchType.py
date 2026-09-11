from enum import Enum


class SearchType(str, Enum):
    SUMMARIES = "SUMMARIES"
    CHUNKS = "CHUNKS"
    RAG_COMPLETION = "RAG_COMPLETION"
    HYBRID_COMPLETION = "HYBRID_COMPLETION"
    TRIPLET_COMPLETION = "TRIPLET_COMPLETION"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    GRAPH_COMPLETION_DECOMPOSITION = "GRAPH_COMPLETION_DECOMPOSITION"
    GRAPH_SUMMARY_COMPLETION = "GRAPH_SUMMARY_COMPLETION"
    CYPHER = "CYPHER"
    NATURAL_LANGUAGE = "NATURAL_LANGUAGE"
    GRAPH_COMPLETION_COT = "GRAPH_COMPLETION_COT"
    GRAPH_COMPLETION_CONTEXT_EXTENSION = "GRAPH_COMPLETION_CONTEXT_EXTENSION"
    FEELING_LUCKY = "FEELING_LUCKY"
    TEMPORAL = "TEMPORAL"
    CODING_RULES = "CODING_RULES"
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"
    AGENTIC_COMPLETION = "AGENTIC_COMPLETION"
    CODE = "CODE"
    GRAPH_REPORT = "GRAPH_REPORT"
    SKILLS = "SKILLS"

    @property
    def required_permissions(self) -> tuple[str, ...]:
        """Dataset permissions a caller must hold on every dataset this search type touches.

        Every type needs ``read``. The types that execute Cypher, whether the caller
        wrote it (CYPHER) or an LLM wrote it from the question (NATURAL_LANGUAGE), also
        need ``write``: nothing restricts the statement to reads, so a read-only share
        must not be enough to run one. ``authorized_search`` enforces this and the
        FEELING_LUCKY selector refuses to route to these types.
        """
        if self in _CYPHER_EXECUTING_SEARCH_TYPES:
            return ("read", "write")
        return ("read",)


_CYPHER_EXECUTING_SEARCH_TYPES = frozenset({SearchType.CYPHER, SearchType.NATURAL_LANGUAGE})
