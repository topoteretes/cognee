from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.shared.logging_utils import get_logger


logger = get_logger("feedback_utils")


def create_retriever(
    retriever_name: str = "graph_completion_cot",
    top_k: int = 20,
    user_prompt_path: str = "graph_context_for_question.txt",
    system_prompt_path: str = "answer_simple_question.txt",
):
    """Factory for retriever instances with configurable top_k and prompt paths."""
    if retriever_name == "graph_completion":
        return GraphCompletionRetriever(
            top_k=top_k,
            save_interaction=False,
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
        )
    if retriever_name == "graph_completion_cot":
        return GraphCompletionCotRetriever(
            top_k=top_k,
            save_interaction=False,
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
        )
    if retriever_name == "graph_completion_context_extension":
        return GraphCompletionContextExtensionRetriever(
            top_k=top_k,
            save_interaction=False,
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
        )
    logger.warning(
        "Unknown retriever, defaulting to graph_completion_cot", retriever=retriever_name
    )
    return GraphCompletionCotRetriever(
        top_k=top_k,
        save_interaction=False,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
    )


def filter_negative_feedback(feedback_nodes):
    """Filter for negative sentiment feedback using precise sentiment classification."""
    return [
        (node_id, props)
        for node_id, props in feedback_nodes
        if (props.get("sentiment", "").casefold() == "negative" or props.get("score", 0) < 0)
    ]
