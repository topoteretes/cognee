"""Cognee-powered agent — single call to the knowledge graph.

Uses cognee.search() which does vector search over graph nodes, retrieves
connected triplets, and generates an LLM answer from the graph context —
all in one call, no agent loop needed.

Supports multiple search types via the ``query_type`` parameter:
- GRAPH_COMPLETION (default) — vector search + graph traversal + LLM answer
- GRAPH_COMPLETION_COT — chain-of-thought reasoning over graph context
- GRAPH_COMPLETION_CONTEXT_EXTENSION — extended context graph retrieval
- TRIPLET_COMPLETION — triplet-based search
- RAG_COMPLETION — traditional RAG with chunks

Token usage is captured by monkey-patching litellm.acompletion so that
every raw LLM response is intercepted before instructor processes it.
"""

import threading
from typing import Dict, List, Optional

import litellm

import cognee
from cognee import SearchType

SYSTEM_PROMPT = (
    "You are a supply chain analyst at ACME Electronics. "
    "Answer the question using the provided knowledge graph context. "
    "Always provide specific IDs, numbers, and names. "
    "If the context contains partial data, answer with what is available "
    "and note what is missing. Show your work for any calculations."
)


# ── Token tracker via litellm.acompletion monkey-patch ────────────────


class _TokenTracker:
    """Thread-safe accumulator for litellm token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.llm_calls = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "llm_calls": self.llm_calls,
            }

    def record(self, response):
        usage = getattr(response, "usage", None)
        if not usage:
            return
        with self._lock:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.llm_calls += 1


_tracker = _TokenTracker()

_original_acompletion = litellm.acompletion


async def _tracking_acompletion(*args, **kwargs):
    response = await _original_acompletion(*args, **kwargs)
    _tracker.record(response)
    return response


litellm.acompletion = _tracking_acompletion


# ── Public API ────────────────────────────────────────────────────────


async def add_to_graph(text: str, node_set: str = "planner_notes") -> None:
    """Ingest new data into the graph and rebuild."""
    await cognee.add(text, node_set=[node_set])
    await cognee.cognify()


async def run(
    question: str,
    history: Optional[List[Dict]] = None,
    query_type: SearchType = SearchType.GRAPH_COMPLETION,
) -> tuple[str, List[str], dict]:
    """Run the Cognee agent.

    Args:
        question:   The question to answer.
        history:    Optional conversation history (unused currently).
        query_type: Cognee SearchType to use (default: GRAPH_COMPLETION).
                    Try SearchType.GRAPH_COMPLETION_COT for chain-of-thought.

    Returns (answer, list_of_tools_called, metrics).
    """
    _tracker.reset()

    results = await cognee.search(
        query_text=question,
        query_type=query_type,
        top_k=50,
        wide_search_top_k=300,
        system_prompt=SYSTEM_PROMPT,
    )

    metrics = _tracker.snapshot()

    if not results:
        return "(no results from knowledge graph)", ["search_knowledge_graph"], metrics

    first = results[0]
    if isinstance(first, dict):
        answer = first.get("search_result", "")
        if isinstance(answer, list) and answer:
            answer = str(answer[0])
        else:
            answer = str(answer)
    else:
        answer = str(first)

    return answer, ["search_knowledge_graph"], metrics
