from typing import Optional, List

from uuid import NAMESPACE_OID, uuid5, UUID
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.engine.models import NodeSet
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.base_feedback import BaseFeedback
from cognee.modules.retrieval.utils.models import CogneeUserFeedback
from cognee.modules.retrieval.utils.models import UserFeedbackEvaluation
from cognee.tasks.storage import add_data_points, index_graph_edges

logger = get_logger("CompletionRetriever")


class UserQAFeedback(BaseFeedback):
    """
    Interface for handling user feedback queries.
    Public methods:
    - get_context(query: str) -> str
    - get_completion(query: str, context: Optional[Any] = None) -> Any
    """

    def __init__(self, last_k: Optional[int] = 1) -> None:
        """Initialize retriever with optional custom prompt paths."""
        self.last_k = last_k

    async def add_feedback(self, feedback_text: str) -> List[str]:
        feedback_sentiment = await LLMGateway.acreate_structured_output(
            text_input=feedback_text,
            system_prompt="You are a sentiment analysis assistant. For each piece of user feedback you receive, return exactly one of: Positive, Negative, or Neutral classification and a corresponding score from -5 (worst negative) to 5 (best positive)",
            response_model=UserFeedbackEvaluation,
        )

        graph_engine = await get_graph_engine()
        last_interaction_ids = await graph_engine.get_last_user_interaction_ids(limit=self.last_k)

        nodeset_name = "UserQAFeedbacks"
        feedbacks_node_set = NodeSet(id=uuid5(NAMESPACE_OID, name=nodeset_name), name=nodeset_name)
        feedback_id = uuid5(NAMESPACE_OID, name=feedback_text)

        cognee_user_feedback = CogneeUserFeedback(
            id=feedback_id,
            feedback=feedback_text,
            sentiment=feedback_sentiment.evaluation.value,
            score=feedback_sentiment.score,
            belongs_to_set=feedbacks_node_set,
        )

        await add_data_points(data_points=[cognee_user_feedback])

        relationships = []
        relationship_name = "gives_feedback_to"
        to_node_ids = []

        for interaction_id in last_interaction_ids:
            target_id_1 = feedback_id
            target_id_2 = UUID(interaction_id)

            if target_id_1 and target_id_2:
                relationships.append(
                    (
                        target_id_1,
                        target_id_2,
                        relationship_name,
                        {
                            "relationship_name": relationship_name,
                            "source_node_id": target_id_1,
                            "target_node_id": target_id_2,
                            "ontology_valid": False,
                        },
                    )
                )
                to_node_ids.append(str(target_id_2))

        if len(relationships) > 0:
            graph_engine = await get_graph_engine()
            await graph_engine.add_edges(relationships)
            await index_graph_edges(relationships)
            await graph_engine.apply_feedback_weight(
                node_ids=to_node_ids, weight=feedback_sentiment.score
            )

        return [feedback_text]
