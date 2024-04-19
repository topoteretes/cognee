import asyncio
import nest_asyncio
import dspy
from cognee.modules.search.vector.search_similarity import search_similarity

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
        context = asyncio.run(search_similarity(question))

        context_text = "\n".join(context)
        print(f"Context: {context_text}")

        with dspy.context(lm = question_answer_llm):
            answer_prediction = self.generate_answer(context = context_text, question = question)
            answer = answer_prediction.answer

            print(f"Question: {question}")
            print(f"Answer: {answer}")

        return dspy.Prediction(context = context_text, answer = answer)
