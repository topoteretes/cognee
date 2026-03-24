"""Simple callstack demo for agentic trace context.

Run:
    uv run python examples/demos/simple_agent_v2/agentic_callstack_demo.py
"""

from __future__ import annotations

from datetime import datetime, timezone

from agentic_context_trace import (
    agentic_trace_root,
    get_current_agent_context_trace,
)


def llm_gateway(prompt: str) -> str:
    """Downstream method that reads context without explicit parameter passing."""
    ctx = get_current_agent_context_trace()
    if ctx is not None and ctx.task_query:
        prompt = f"Task query: {ctx.task_query}\\n\\n{prompt}"
    return f"LLM_RESPONSE(to={prompt!r})"


def build_answer(user_question: str) -> str:
    """Another downstream method in the call stack."""
    return llm_gateway(prompt=user_question)


@agentic_trace_root(with_memory=True, task_query="Explain in 2 short bullet points.")
def run_agent_flow(user_question: str) -> str:
    """Top-level entrypoint creating the context object."""
    return build_answer(user_question)


if __name__ == "__main__":
    result, ctx = run_agent_flow("How does cognee retrieval work?")
    created_at_iso = datetime.fromtimestamp(ctx.created_at / 1000, tz=timezone.utc).isoformat()

    print(result)
    print(f"origin={ctx.origin_function}")
    print(f"created_at={created_at_iso}")
    print(f"task_query={ctx.task_query}")
    print(f"with_memory={ctx.with_memory}")
