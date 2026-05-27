from cognee.infrastructure.llm.LLMGateway import LLMGateway
from pydantic import BaseModel

DECOMPOSE_PROMPT = """Decompose the hypothesis into exactly one premise and one conclusion.

Instructions:
1. Identify the single statement that must be assumed for the hypothesis to hold. This is the premise.
2. Identify the single statement that the hypothesis predicts or claims when the premise holds. This is the conclusion.
3. Keep each component concise, factual, and self-contained. Do not list multiple premises or conclusions.

Examples:
- Hypothesis: Module alpha alone solves Problem A.
  Premise: Problem A inputs are routed through module alpha.
  Conclusion: The pipeline output matches Problem A targets.
- Hypothesis: Problem A is solved by a three-module cascade.
  Premise: Alpha feeds beta and beta feeds gamma in the active pipeline.
  Conclusion: The final stage produces the Problem A output.
- Hypothesis: X affects Y through Z.
  Premise: X changes and pathway Z is active.
  Conclusion: Y changes via Z.

Output only premise and conclusion text."""


class Decomposition(BaseModel):
    premise: str
    conclusion: str


async def decompose_hypothesis_text(hypothesis: str) -> Decomposition:
    """Decompose a hypothesis string into premise and conclusion."""
    response = await LLMGateway.acreate_structured_output(
        text_input=f"Hypothesis:\n{hypothesis}",
        system_prompt=DECOMPOSE_PROMPT,
        response_model=Decomposition,
    )
    return Decomposition(premise=response.premise.strip(), conclusion=response.conclusion.strip())
