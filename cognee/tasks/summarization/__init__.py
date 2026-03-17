"""
LLM-based summarization tasks.

This module provides tasks for generating structured summaries from
document chunks and code graph nodes. It produces DataPoint-based summary
objects that can be indexed and integrated into the knowledge graph.
"""

from .summarize_code import summarize_code
from .summarize_text import summarize_text
