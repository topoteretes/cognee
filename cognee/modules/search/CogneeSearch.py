import asyncio
import dspy
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client, GraphDBType
from cognee.modules.search.vector.search_similarity import search_similarity

import nest_asyncio
nest_asyncio.apply()

class AnswerFromContext(dspy.Signature):
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc = "Context to use for answer generation.")
    answer: str = dspy.OutputField()

question_answer_llm = dspy.OpenAI(model = "gpt-3.5-turbo-instruct")

class CogneeSearch(dspy.Module):
    def __init__(self, ):
        super().__init__()
        self.generate_answer = dspy.TypedChainOfThought(AnswerFromContext)

    def forward(self, question):
        loop = asyncio.get_running_loop()

        graph_client = loop.run_until_complete(get_graph_client(GraphDBType.NETWORKX))
        graph = graph_client.graph

        context = loop.run_until_complete(search_similarity(question, graph))

        context_text = "\n".join(context)
        print(f"Context: {context_text}")

        with dspy.context(lm = question_answer_llm):
            answer_prediction = self.generate_answer(context = context_text, question = question)
            answer = answer_prediction.answer

            print(f"Question: {question}")
            print(f"Answer: {answer}")

        return dspy.Prediction(context = context_text, answer = answer)
