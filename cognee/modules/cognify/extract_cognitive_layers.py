from typing import List
import dspy

class LayersFromText(dspy.Signature):
    """
    Instructions:
    You are tasked with analyzing data in order to extract meaningful layers of information
    that will contribute to constructing a detailed multilayer network or knowledge graph.
    Consider the unique characteristics and inherent properties of the data at hand.

    VERY IMPORTANT:
    The context and domain you are working in is defined by the "category" input.
    The content category, defined by "category" input, should play a major role in how you decompose into layers.
    """

    text: str = dspy.InputField()
    text_category: str = dspy.InputField(desc = "Category in which the text belongs.")
    cognitive_layers: List[str] = dspy.OutputField(desc = "JSON array of cognitive layers.")


class ExtractCognitiveLayers(dspy.Module):
    def __init__(self, lm = dspy.OpenAI(
        model = "gpt-3.5-turbo",
        max_tokens = 4096
    )):
        super().__init__()
        self.lm = lm
        self.extract_cognitive_layers_from_text = dspy.TypedChainOfThought(LayersFromText)

    def forward(self, text: str, category: str):
        with dspy.context(lm = self.lm):
            return self.extract_cognitive_layers_from_text(
                text = text,
                text_category = category
            ).cognitive_layers
