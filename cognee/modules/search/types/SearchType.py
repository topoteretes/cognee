from enum import Enum


class SearchType(Enum):
    SUMMARIES = "SUMMARIES"
    INSIGHTS = "INSIGHTS"
    CHUNKS = "CHUNKS"
    COMPLETION = "COMPLETION"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    CODE = "CODE"
