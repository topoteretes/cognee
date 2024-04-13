from typing import List
import dspy
from cognee.shared.data_models import TextContent

class CategoriesFromText(dspy.Signature):
    """
    Instructions:
    You are a classification engine and should classify content.
    Make sure to use one of the existing classification options nad not invent your own.
    """

    text: str = dspy.InputField()
    categories: List[TextContent] = dspy.OutputField(desc = "JSON array of categories in which the text belongs.")


class ExtractCategories(dspy.Module):
    def __init__(self, lm = dspy.OpenAI(
        model = "gpt-3.5-turbo",
        max_tokens = 4096
    )):
        super().__init__()
        self.lm = lm
        self.extract_categories_from_text = dspy.TypedPredictor(CategoriesFromText)
    
    def forward(self, text: str):
        with dspy.context(lm = self.lm):
            return self.extract_categories_from_text(text = text).categories
